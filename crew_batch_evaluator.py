import pandas as pd
import time
from tqdm import tqdm
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Load ticket dataset
df = pd.read_csv("it_tickets_500.csv")
tickets = df["ticket"].tolist()
true_labels = df["category"].tolist()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Define Classifier Agent only (we'll test classification performance)
classifier = Agent(
    role="Issue Classifier",
    goal="Classify the IT issue into one of the following categories: Network, Software, Hardware, Access, Other. Respond only with the category.",
    backstory="A smart LLM trained to categorize support tickets into fixed categories.",
    llm=llm,
    verbose=False
)

correct = 0
total_time = 0
misclassified = []

print("🧪 Evaluating Crew AI Classifier on 500 Tickets...\n")

for i, (ticket, true_cat) in enumerate(tqdm(zip(tickets, true_labels), total=len(tickets))):
    start = time.time()

    task = Task(
        agent=classifier,
        description=f"Classify this IT ticket: '{ticket}'. Respond only with one of: Network, Software, Hardware, Access, Other."
    )

    crew = Crew(
        agents=[classifier],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()
    duration = time.time() - start
    total_time += duration

    predicted = result.strip().split()[0]  # Simple cleanup
    if predicted.lower() == true_cat.lower():
        correct += 1
    else:
        misclassified.append((ticket, true_cat, predicted))

# Final Stats
accuracy = (correct / len(tickets)) * 100
avg_time = total_time / len(tickets)

print("\n✅ Evaluation Complete!")
print(f"📊 Accuracy: {accuracy:.2f}%")
print(f"⏱️ Avg. Time per Ticket: {avg_time:.2f} seconds")
print(f"❌ Misclassified Samples (showing 5):")
for ticket, true_cat, pred in misclassified[:5]:
    print(f" - Ticket: '{ticket}' | True: {true_cat} | Predicted: {pred}")
