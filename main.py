from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from note_engine import note_engine
from pdf import Resume_engine
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine import PandasQueryEngine
from prompt import new_prompt, instruction_str, context

load_dotenv()
resume_data = os.path.join("data", "my_data.csv")
resume_df = pd.read_csv(resume_data)
resume_query_engine = PandasQueryEngine(df=resume_df, verbose=True, instruction_str=instruction_str)
resume_query_engine.update_prompts({"pandas_prompt": new_prompt})
#mental_query_engine.query("i am feeling great")

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=resume_query_engine,
        metadata=ToolMetadata(
            name="multiple_Resume_data",
            description="this gives information about all resume",
        ),
    ),
    QueryEngineTool(
        query_engine=Resume_engine,
        metadata=ToolMetadata(
            name="Data_Science",
            description="this gives detailed information about Data Science roles and job",
        ),
    ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
    
    