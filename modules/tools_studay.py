"""Toolkit for interacting with an SQL database."""
from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from tools.tool_for_agent import *
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool

class SQLDataAgentToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )

        
        info_sql_plot_tool = (
            "You are an agent designed to write python code to make and save the graph in order to answer questions."
            f"Input to this tool is the sql data which is {query_sql_database_tool}"
            "Use matplotlib or pandas or seaborn to generate the graph or tables"
            "Output is the plot pnrrjg file that plots visual graph or table"
            "output example: png file"
        )

        sql_plot_tool = PythonREPLTool(description=info_sql_plot_tool)

        # info_sql_execute_plot_tool = (
        #     "You are an agent designed to execute python code to plot the graph in order to answer questions."
        #     f"Input to this tool is the sql data which is {sql_plot_tool}"
        #     "just execute the python code"
        # )
        
        # sql_execute_plot_tool =PythonAstREPLTool(description=info_sql_execute_plot_tool)
        
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            sql_plot_tool
        ]
        

    def get_context(self) -> dict:
        """Return db context that you may want in agent prompt."""
        return self.db.get_context()
