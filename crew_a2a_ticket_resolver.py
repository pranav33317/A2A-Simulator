from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import time

# Sample tickets (can be replaced by reading from a CSV)
tickets = [
    "Cannot connect to the VPN since morning. Getting authentication error.",
    "Outlook crashes on startup with error 0x80042108.",
    "Need access to GitLab repo for urgent deployment.",
    "Laptop battery drains quickly, even on sleep mode.",
    "Wi-Fi disconnects frequently while attending Zoom meetings."
]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Define agents
ticket_reader = Agent(
    role='Ticket Reader',
    goal='Read user complaint and summarize the issue clearly',
    backstory='An assistant that processes incoming user tickets',
    llm=llm,
    verbose=True
)

classifier = Agent(
    role='Issue Classifier',
    goal='Classify the IT issue into categories: Network, Software, Hardware, Access, Other. Respond only with the category.',
    backstory='A smart LLM trained to categorize support tickets into fixed categories.',
    llm=llm,
    verbose=True
)

resolver = Agent(
    role='Issue Resolver',
    goal='Suggest 3–5 clear and concise steps to resolve the issue based on its classification.',
    backstory='An automated expert with troubleshooting knowledge.',
    llm=llm,
    verbose=True
)

dispatcher = Agent(
    role='Dispatcher',
    goal='Prepare a professional response email to the user with the resolution steps.',
    backstory='An assistant that sends clear and courteous replies to users.',
    llm=llm,
    verbose=True
)

# Run the pipeline for each ticket
for i, ticket in enumerate(tickets):
    print(f"\n🚀 Processing Ticket #{i+1}: {ticket}")
    start_time = time.time()

    # Define the pipeline tasks
    task1 = Task(agent=ticket_reader, description=f"Read this IT support ticket and summarize it: '{ticket}'")
    task2 = Task(agent=classifier, description="Classify the summarized issue as one of: Network, Software, Hardware, Access, Other.")
    task3 = Task(agent=resolver, description="Generate 3–5 resolution steps based on the classification and ticket context.")
    task4 = Task(agent=dispatcher, description="Write a professional response email containing the resolution steps.")

    # Create crew
    crew = Crew(
        agents=[ticket_reader, classifier, resolver, dispatcher],
        tasks=[task1, task2, task3, task4],
        verbose=True
    )

    # Run it
    final_output = crew.kickoff()
    duration = round(time.time() - start_time, 2)

    print(f"\n⏱️ Completed in {duration} seconds")
    print(f"📧 Final Output:\n{final_output}")
