import os
import re
import shutil
import zipfile

import pandas as pd
import requests
import s3fs
from loguru import logger

from src.components.data_processing.data_loader import DataUtils
from src.components.utils.config import DownloadDataConfig
from src.components.utils.read_config import ConfigReader


class UtilsDownloader:

    def __init__(self):
        """"""
        pass

    def load_file_system(self, endpoint_url=None):
        """Load S3 filesystem"""
        if endpoint_url:
            self.fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint_url})
        else:
            self.fs = s3fs.S3FileSystem()
        logger.info(f"S3 filesystem loaded with endpoint: {endpoint_url or 'default'}")

    def create_directory(self, directory_path):
        """Create directory - works for both local and S3"""
        if self.use_s3:
            # S3 doesn't need explicit directory creation
            pass
        else:
            os.makedirs(directory_path, exist_ok=True)

    def file_exists(self, file_path):
        """Check if file exists - works for both local and S3"""
        if self.use_s3:
            return self.fs.exists(file_path)
        else:
            return os.path.exists(file_path)

    def remove_file(self, file_path):
        """Remove file - works for both local and S3"""
        if self.use_s3:
            if self.fs.exists(file_path):
                self.fs.rm(file_path)
        else:
            if os.path.exists(file_path):
                os.remove(file_path)

    def remove_directory(self, dir_path):
        """Remove directory - works for both local and S3"""
        if self.use_s3:
            if self.fs.exists(dir_path):
                self.fs.rm(dir_path, recursive=True)
        else:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    def list_files(self, directory_path):
        """List files in directory - works for both local and S3"""
        if self.use_s3:
            if self.fs.exists(directory_path):
                return self.fs.ls(directory_path, detail=False)
            return []
        else:
            if os.path.exists(directory_path):
                return [
                    os.path.join(directory_path, f) for f in os.listdir(directory_path)
                ]
            return []


class DataDownloader(UtilsDownloader):
    """Data downloader local / S3"""

    def __init__(self):
        """Initialize the pipeline with configuration file."""
        super().__init__()
        self.config = ConfigReader._read_config(
            "config/download_data.json", DownloadDataConfig
        )
        self.fs = (
            DataUtils._create_fs()
            if DataUtils._detect_s3(self.config.data_path)
            else os
        )
        self.use_s3 = DataUtils._detect_s3(self.config.data_path)

        # if self.use_s3:
        #     self.load_file_system(self.config["s3_endpoint"])
        #     self.config.data_path = (
        #         f"s3://{self.config['bucket']}/{self.config['s3_data_path']}"
        #     )
        # else:
        #     self.config.data_path = self.config["local_data_path"]

        logger.debug(f"Using data path: {self.config.data_path}")

    def format_content(self, content):
        # Type error in Age_csv
        content = content.replace("Age_csp.zip", "Age_csp.zip\n")
        content = content.replace("csv.zip", "csv.zip\n")
        content = re.sub(r"https", r"\nhttps", content)
        lines = [line for line in content.splitlines() if "zip" in line]
        lines = [re.sub(r"\.zip.*$", ".zip", line) for line in lines]
        lines = [line for line in lines if (("csv.zip" in line) or ("csp.zip" in line))]
        return lines

    def make_right_directory(self, file_url, base_dir):
        name_file = file_url.split("/")[-1].split("_")[0]

        # Define election type mappings
        election_types = {
            "pre": ("presidentiel", 4, 8),  # (folder_name, year_start, year_end)
            "ref": ("referendum", 3, 7),
            "leg": ("legislative", 3, 7),
        }

        prefix = name_file[:3]

        if prefix in election_types:
            folder_name, start_idx, end_idx = election_types[prefix]
            year = name_file[start_idx:end_idx]
            if name_file in ["leg1871juil", "leg1946Nov"]:
                year = str(int(year) + 1)
                name_file = re.sub(r"(juil|Nov$", "", name_file)
                logger.warning(f"Election {name_file} is given the year {year}")
            if self.use_s3:
                output_dir = f"{base_dir.rstrip('/')}/elections/{folder_name}/{year}"
            else:
                output_dir = os.path.join(base_dir, "elections", folder_name, year)
        else:
            output_dir = base_dir

        # Create output directory
        self.create_directory(output_dir)
        return output_dir, name_file

    def fetch_website(self, url, base_dir):
        # Fetch links on website
        response = requests.get(url)

        if response.status_code == 200:
            content = response.text
            output = self.format_content(content)

            for file_url in output[::-1]:
                # Create directory
                output_dir, name_file = self.make_right_directory(file_url, base_dir)

                # Download ZIP file
                logger.info(f"Downloading {name_file}...")
                response = requests.get(file_url)

                if response.status_code == 200:
                    # Zip file path
                    if self.use_s3:
                        zip_file_path = f"{output_dir}/{name_file}.zip"
                    else:
                        zip_file_path = os.path.join(output_dir, f"{name_file}.zip")

                    # Save the downloaded zip file
                    if self.use_s3:
                        with self.fs.open(zip_file_path, "wb") as f:
                            f.write(response.content)
                    else:
                        with open(zip_file_path, "wb") as f:
                            f.write(response.content)

                    # For S3, we need to download zip locally first to extract
                    if self.use_s3:
                        # Download zip to temp local file for extraction
                        temp_zip = f"/tmp/{name_file}.zip"
                        with open(temp_zip, "wb") as f:
                            f.write(response.content)

                        # Extract to temp directory
                        temp_extract_dir = f"/tmp/{name_file}_extract"
                        os.makedirs(temp_extract_dir, exist_ok=True)

                        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                            zip_ref.extractall(temp_extract_dir)

                        # Process extracted files and upload to S3
                        for root, subfolders, files in os.walk(temp_extract_dir):
                            if len(subfolders) > 0:
                                for subfolder in subfolders:
                                    subfolder_path = os.path.join(root, subfolder)

                                    for root_, subfolders_, files_ in os.walk(
                                        subfolder_path
                                    ):
                                        for filename in files_:
                                            local_file_path = os.path.join(
                                                root_, filename
                                            )

                                            if filename.endswith(".rtf"):
                                                continue
                                            elif filename.endswith(".csv"):

                                                # Read and process CSV
                                                try:
                                                    df = pd.read_csv(
                                                        local_file_path,
                                                        low_memory=False,
                                                        encoding="unicode_escape",
                                                        on_bad_lines="warn",
                                                    )
                                                except UnicodeDecodeError:
                                                    pass

                                                object_cols = df.select_dtypes(
                                                    include=["object"]
                                                ).columns
                                                df[object_cols] = df[
                                                    object_cols
                                                ].astype(str)

                                                # Save as parquet to S3
                                                s3_parquet_path = f"{output_dir}/{subfolder}/{filename.replace('.csv', '.parquet')}"

                                                with self.fs.open(
                                                    s3_parquet_path, "wb"
                                                ) as f:
                                                    df.to_parquet(f)

                        # Clean up temp files
                        os.remove(temp_zip)
                        shutil.rmtree(temp_extract_dir)

                        macosx_path = os.path.join(output_dir, "__MACOSX")
                        self.remove_directory(macosx_path)

                        # Remove zip from S3
                        self.remove_file(zip_file_path)

                    else:
                        # Local processing (original logic)
                        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                            zip_ref.extractall(output_dir)

                        # Remove ZIP
                        self.remove_file(zip_file_path)

                        # Remove __MACOSX
                        macosx_path = os.path.join(output_dir, "__MACOSX")
                        self.remove_directory(macosx_path)

                        # Convert to parquet and remove rtf
                        for root, subfolders, files in os.walk(output_dir):
                            if len(subfolders) > 0:
                                for subfolder in subfolders:
                                    subfolder_path = os.path.join(root, subfolder)

                                    for root_, subfolders_, files_ in os.walk(
                                        subfolder_path
                                    ):
                                        for filename in files_:
                                            file_path = os.path.join(root_, filename)
                                            if filename.endswith(".rtf"):

                                                self.remove_file(file_path)
                                            elif filename.endswith(".csv"):

                                                csv_path = file_path
                                                df = pd.read_csv(
                                                    csv_path,
                                                    low_memory=False,
                                                    encoding="unicode_escape",
                                                    on_bad_lines="warn",
                                                )
                                                object_cols = df.select_dtypes(
                                                    include=["object"]
                                                ).columns
                                                df[object_cols] = df[
                                                    object_cols
                                                ].astype(str)
                                                parquet_path = os.path.join(
                                                    root_,
                                                    filename.replace(
                                                        ".csv", ".parquet"
                                                    ),
                                                )
                                                df.to_parquet(parquet_path)
                                                self.remove_file(csv_path)
                else:
                    logger.error(f"Unable to download file {file_url}")
        else:
            logger.error("Unable to fetch website")

        return output


if __name__ == "__main__":
    # Example usage with S3
    dd = DataDownloader()
    dd.fetch_website(url=dd.config["url"], base_dir=dd.data_path)
