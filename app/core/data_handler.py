import s3fs
import streamlit as st

class FileSystem:

    def __init__(self, client_kwargs, key, secret):
        self.client_kwargs = client_kwargs
        self.key = key
        self.secret = secret

    def load_fs(self):
        self.fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': self.client_kwargs},
            key=self.key,
            secret=self.secret,
        )

    def get_fs(self):
        return self.fs




class AppData:
    pass