from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os

# Set up the OpenAI API key from environment variables
# Ensure OPENAI_API_KEY is set in your environment or .env file
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" 

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Define Agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and blockchain',
    backstory='You are a seasoned analyst with a knack for identifying emerging trends and disruptive technologies.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Technical Content Writer',
    goal='Produce engaging and informative articles on AI and blockchain innovations',
    backstory='You are a skilled writer, able to translate complex technical concepts into clear and compelling narratives.',
    llm=llm,
    verbose=True,
    allow_delegation=True
)

# Define Tasks
research_task = Task(
    description='Identify the top 3 recent advancements in AI and their potential impact on various industries.',
    agent=researcher,
    expected_output='A detailed report outlining 3 key AI advancements and their industry implications.'
)

write_task = Task(
    description='Write a blog post (500 words) summarizing the research findings, focusing on clarity and engagement.',
    agent=writer,
    expected_output='A 500-word blog post suitable for a general tech audience.'
)

# Form the Crew
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=2
)

# Kick off the workflow
if __name__ == "__main__":
    print("## Starting Multi-Agent Workflow ##")
    result = tech_crew.kickoff()
    print("\n## Workflow Finished ##")
    print(result)
