from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv
import os

load_dotenv()

scraper = ScrapeWebsiteTool()

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GROQ_API_KEY")
)

web_scraping_agent = Agent(
    role="Web Scraping Expert",
    goal="Scrape the website and extract relevant information",
    backstory="You are skilled at extracting data from websites efficiently",
    tools=[scraper],
    llm=llm
)

web_scraping_task = Task(
    description="Scrape the website and extract relevant information from https://www.nasa.gov/",
    expected_output="Structured data extracted from the website",
    agent=web_scraping_agent
)

crew = Crew(
    agents=[web_scraping_agent],
    tasks=[web_scraping_task],
    verbose=False
)

result = crew.kickoff()
print(result)
