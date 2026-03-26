from typing import Any
from pymongo import MongoClient
from pymongo.database import Database
from sshtunnel import SSHTunnelForwarder
import os

class MongoConnection:
    def __init__(self):
        # if os.getenv("USE_SSH_TUNNEL_FORWARDING") == "1":
        #     print("entrou ssh")
        #     self._tunnel = self._get_ssh_tunnel()
        #     print("criou ssh")
        #     self._tunnel.start()

        mongo_db_port = os.getenv("MONGO_DB_PORT")
        if mongo_db_port is None:
            raise ValueError("porta não definida")

        self._client: MongoClient[Any] = MongoClient('mongodb://%s:%s/' % (os.getenv("MONGO_DB_HOST"), int(mongo_db_port)))

    def get_instagram_connection(self) -> Database[Any]:
        db_name: str | None = os.getenv("MONGO_DB_INSTAGRAM_DATABASE")
        if db_name is None:
            raise ValueError("MONGO_DB_INSTAGRAM_DATABASE não definida")
        
        return self._client[db_name]

    def close_connections(self):
        # if self._tunnel:
        #     self._tunnel.close()
        
        if self._client:
            self._client.close()

    def _get_ssh_tunnel(self):
        ssh_port = os.getenv("SSH_PORT")
        mongo_db_port = os.getenv("MONGO_DB_PORT")

        if ssh_port is None or mongo_db_port is None:
            raise ValueError("porta não definida")

        return SSHTunnelForwarder(
            (os.getenv("SSH_HOST"), int(ssh_port)),
            ssh_username=os.getenv("SSH_USERNAME"),
            ssh_password=os.getenv("SSH_PASSWORD"),
            remote_bind_address=(os.getenv("MONGO_DB_HOST"), int(mongo_db_port)),
        )