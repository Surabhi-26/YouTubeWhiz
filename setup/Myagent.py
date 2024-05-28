from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOpenAI(
    model = "crewai-llama3",
    base_url = "http://localhost:11434/v1")


from crewai_tools import YoutubeChannelSearchTool

tool = YoutubeChannelSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", 
            config=dict(
                model="llama3",
               
            ),
        ),
        embedder=dict(
            provider="huggingface", 
            config=dict(
                model="sentence-transformers/msmarco-distilbert-base-v4"
            ),
        ),
    ),
    youtube_channel_handle='@youtubechannel'
)

agentyt=Agent(
    role='Python Tutorial YouTube videos Searcher',
    goal='To fetch relevant youtube videos',
    backstory="""You are a video searcher specialist and content recommender. You are responsible to search the appropriate video that satisfies the user's need""",
    tools=[tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

searchyt=Task(
    description="Search the youtube videos of python tutorials",
    expected_output="Provide the links of youtube videos of python tutorials",
    agent=agentyt
)

crew=Crew(
    agents=[agentyt],
    tasks=[searchyt],
    verbose=2
)

result=crew.kickoff()
print(result)