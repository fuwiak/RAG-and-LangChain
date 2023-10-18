from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from dotenv import load_dotenv, dotenv_values
import openai
import os
import pandas as pd
from typing import List, Any, Callable


def load_openai_key_from_env(filepath=".env"):
    config = dotenv_values(filepath)
    openai_api_key = config['open_ai_key']
    set_openai_key(openai_api_key)


def set_openai_key(api_key):
    openai.api_key = api_key
    os.environ['OPENAI_API_KEY'] = api_key


data = {
    'Group': [1, 2, 3],
    'Group_1': [
        "@JamesSmith James, great job on the project! Now imagine if everyone in the protest group, soldiers, pilots, and others were this efficient. The first to respond were the elite forces from Spearhead Division.",
        "@LauraMiles Laura, please apologize to the entire community rather than playing the hero. Soldiers and the entire protest group did much more.",
        'What about the "Brothers in Arms" group? Suddenly they agree on the new policy?'
    ],
    'Group_2': [
        "Hey everyone!👋\n\nStarting off a new week with some exciting news. Keep an eye on our Instagram for an exclusive offer.🎁👀\n\n👉Only for our Instagram followers.\n\n📢🔴Share our...",
        "BREAKING: The mock test is now LIVE🟥 https://drive.google.com/file/d/randomlink/view?usp=sharing\n\nDownload from our blog or check our Instagram profile. #mocktest...",
        "Hello everyone!👋 We've got a new section in our Instagram stories. 🔴Icon quizzes and shortcut challenges. What do you think? 🔜Stay tuned, more surprises coming..."
    ],
    'Group_3': [
        "Go team!👮🏻‍♂️🚔 #newproject #teamwork #officechallenges #trending #viral #forourfollowers #behindthescenes #worklife #colleagues #bestteam",
        "With renowned journalist and friend, David Foster. Presented a talk about the life of Beethoven. Here with my buddy, Professor Johnson. Recommend this talk to all institutions.",
        "Study hard, play hard!\n\n#worklife #balance #challengeaccepted #goals #unilife #pushthelimits #determined #focused #keepgoing"
    ]
}

df = pd.DataFrame(data)


def setup_chain(selected_columns: List[str]) -> Callable:

    def get_context_from_columns() -> str:
        return " ".join(selected_columns)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    vectorstore = FAISS.from_texts(selected_columns, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


def summarize_all_posts(df, start_col: int, end_col: int) -> str:
    output = " "
    for col in df.columns[start_col:end_col + 1]:
        selected_columns = df[col].tolist()
        chain_instance = setup_chain(selected_columns)
        response = chain_instance.invoke("Summarize text: ")
        output += response + "\n"
    return output


def print_chain_results(df, question: str, start_col: int, end_col: int) -> None:
    for col in df.columns[start_col:end_col + 1]:
        output = summarize_all_posts(df, start_col, end_col).split("\n")
        answer_chain = setup_chain(output)
        answer = answer_chain.invoke(question)

        print(f"Question: {question}")
        print(f"Number of group: {col}")
        print(f"Answer: {answer}\n")


load_dotenv()
load_openai_key_from_env()



question = "What is most popular topic in text?"

# Example usage
start_col_index = 1
end_col_index = 3
print_chain_results(df, question, start_col_index, end_col_index)
