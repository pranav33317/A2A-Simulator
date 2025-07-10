import streamlit as st
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenAI key (or use st.secrets)
os.environ["OPENAI_API_KEY"] = "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"


llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Define agents
ticket_reader = Agent(
    role='Ticket Reader',
    goal='Read and summarize the issue clearly',
    backstory='An assistant that processes incoming user tickets',
    llm=llm,
    verbose=False
)

classifier = Agent(
    role='Issue Classifier',
    goal='Classify the IT issue into: Network, Software, Hardware, Access, Other. Reply with the category only.',
    backstory='Categorizes IT issues for automated routing.',
    llm=llm,
    verbose=False
)

resolver = Agent(
    role='Issue Resolver',
    goal='Suggest 3–5 steps to resolve the issue based on its classification.',
    backstory='An expert LLM with troubleshooting knowledge.',
    llm=llm,
    verbose=False
)

dispatcher = Agent(
    role='Dispatcher',
    goal='Write a professional response email with the resolution steps.',
    backstory='Replies to users on behalf of the IT helpdesk.',
    llm=llm,
    verbose=False
)

# Streamlit UI
st.title("🧠 IT Ticket Auto-Resolver (Crew AI)")

ticket_input = st.text_area("Paste your IT support ticket:")

if st.button("Run LLM Pipeline"):
    if not ticket_input.strip():
        st.warning("Please enter a ticket.")
    else:
        with st.spinner("Processing with multi-agent LLM..."):
            # Define tasks
            task1 = Task(agent=ticket_reader, description=f"Summarize this IT support ticket: '{ticket_input}'")
            task2 = Task(agent=classifier, description="Classify the issue as: Network, Software, Hardware, Access, Other.")
            task3 = Task(agent=resolver, description="Suggest 3–5 steps to resolve the issue.")
            task4 = Task(agent=dispatcher, description="Write a response email with the resolution.")

            crew = Crew(
                agents=[ticket_reader, classifier, resolver, dispatcher],
                tasks=[task1, task2, task3, task4],
                verbose=False
            )

            result = crew.kickoff()

        # Display results
        st.success("✅ Ticket Processed!")
        st.subheader("📋 Summary")
        st.write(task1.output.strip())

        st.subheader("🗂 Predicted Category")
        st.write(task2.output.strip())

        st.subheader("🛠 Resolution Steps")
        st.write(task3.output.strip())

        st.subheader("📧 Final Response Email")
        st.write(task4.output.strip())
