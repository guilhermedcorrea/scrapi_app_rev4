
from itemadapter import ItemAdapter
import pyodbc

class DatabasePipeline:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            connection_string=crawler.settings.get('DATABASE_CONNECTION')
        )

    def open_spider(self, spider):
        self.connection = pyodbc.connect(self.connection_string)
        self.cursor = self.connection.cursor()

    def close_spider(self, spider):
        self.connection.close()

    def process_item(self, item, spider):
        query = "INSERT INTO your_table (column1, column2, column3) VALUES (?, ?, ?)"
        values = (item['value1'], item['value2'], item['value3'])
        
        self.cursor.execute(query, values)
        self.connection.commit()
        
        return item
