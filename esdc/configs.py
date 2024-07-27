from dataclasses import dataclass, field
from platformdirs import PlatformDirs


@dataclass(frozen=True)
class Config:
    APP_NAME: str = "esdc"
    APP_AUTHOR: str = "skk"
    BASE_API_URL_V2: str = "https://esdc.skkmigas.go.id/"

    @classmethod
    def get_db_path(cls):
        dirs = PlatformDirs(appname=cls.APP_NAME, appauthor=cls.APP_AUTHOR)
        return dirs.user_data_path
