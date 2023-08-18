from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    name = Column(String)


engine = create_engine('mssql+pyodbc://sa:123@DESKTOP-LRA2H5S/DEV?driver=ODBC+Driver+17+for+SQL+Server')
Session = sessionmaker(bind=engine)

def add_item(name):
    item = Item(name=name)
    session = Session()
    session.add(item)
    session.commit()
    session.close()
