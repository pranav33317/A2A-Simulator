import pandas as pd
import time
from tqdm import tqdm
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Load ticket CSV
df = pd.read_csv("it_tickets_500.csv")  # Make sure this CSV is in the same directory
tickets = df["ticket"].tolist()
true_labels = df["category"].tolist()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Define agents
ticket_reader = Agent(
    role='Ticket Reader',
    goal='Read user complaint and summarize the issue clearly',
    backstory='An assistant that processes incoming user tickets',
    llm=llm,
    verbose=False
)

classifier = Agent(
    role='Issue Classifier',
    goal='Classify the IT issue into categories: Network, Software, Hardware, Access, Other. Respond only with the category.',
    backstory='A smart LLM trained to categorize support tickets into fixed categories.',
    llm=llm,
    verbose=False
)

resolver = Agent(
    role='Issue Resolver',
    goal='Suggest 3–5 clear and concise steps to resolve the issue based on its classification.',
    backstory='An automated expert with troubleshooting knowledge.',
    llm=llm,
    verbose=False
)

dispatcher = Agent(
    role='Dispatcher',
    goal='Prepare a professional response email to the user with the resolution steps.',
    backstory='An assistant that sends clear and courteous replies to users.',
    llm=llm,
    verbose=False
)

# Initialize stats
correct = 0
total_time = 0
misclassified = []

print("🧪 Running Full LLM Pipeline on 500 Tickets...\n")

for i, (ticket, true_cat) in enumerate(tqdm(zip(tickets, true_labels), total=len(tickets))):
    start = time.time()

    # Define the 4-step LLM pipeline
    task1 = Task(agent=ticket_reader, description=f"Read and summarize this IT support ticket: '{ticket}'")
    task2 = Task(agent=classifier, description="Classify the summarized issue as one of: Network, Software, Hardware, Access, Other.")
    task3 = Task(agent=resolver, description="Generate 3–5 resolution steps based on the classification and ticket context.")
    task4 = Task(agent=dispatcher, description="Write a professional response email containing the resolution steps.")

    crew = Crew(
        agents=[ticket_reader, classifier, resolver, dispatcher],
        tasks=[task1, task2, task3, task4],
        verbose=False
    )

    try:
        result = crew.kickoff()
        duration = time.time() - start
        total_time += duration

        # Use output from task2 (classifier) for evaluation
        predicted = task2.output.strip().split()[0]
        if predicted.lower() == true_cat.lower():
            correct += 1
        else:
            misclassified.append((ticket, true_cat, predicted))

    except Exception as e:
        print(f"❌ Error on ticket {i+1}: {e}")
        misclassified.append((ticket, true_cat, "Error"))

# Final statistics
accuracy = (correct / len(tickets)) * 100
avg_time = total_time / len(tickets)

print("\n✅ Evaluation Complete!")
print(f"📊 Accuracy: {accuracy:.2f}%")
print(f"⏱️ Avg. Time per Ticket: {avg_time:.2f} seconds")
print(f"❌ Misclassified Samples (showing up to 5):")
for ticket, true_cat, pred in misclassified[:5]:
    print(f" - Ticket: '{ticket}' | True: {true_cat} | Predicted: {pred}")
