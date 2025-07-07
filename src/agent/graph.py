"""This file contains the graph for the medical agent."""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, MessagesState, StateGraph

llm = init_chat_model("openai:gpt-4o")

context_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """As a medical professional, you will provide objective clinical information based on the available patient data. The user initially has no background information, so provide only the specific details they request.

You have this information:
- Question: {question}
- Clinical Context: {context}
- Answer Choices: {options}
- Lab Results: {labResults}
- Blood Pressure: {bloodPresure}
- Pulse: {pulse}
- Respirations: {respirations}
- Physical Examination: {physicalExamination}
- Temperature: {temperature}
- History: {history}
- Demographics: {demographics}
- Family History: {familyHistory}
- Allergies: {allergies}
- History: {history}
- Medications: {medications}





Example Response 1:
User: "Can you tell me about the patient's symptoms?"
Assistant: "The patient is a 13 year old girl who feels well but is worried about not starting puberty yet. Physical examination shows a lack of secondary sexual characteristics and a blind vagina is noted on pelvic examination."

Example Response 2:
User: "What are the patient's vital signs?"
Assistant: "The patient's vital signs are:
- Temperature: 37.0°C
- Blood Pressure: 152/91 mmHg  
- Respiratory Rate: 18/min"

Instructions:
1. Only provide information that is explicitly given in the patient data
2. If asked about information you don't have, respond with "I don't have access to that information"
3. Keep responses focused on objective clinical findings
4. Do not speculate or make assumptions
5. If asked about the answer or any hints, respond with "Please submit your answer first before asking for explanations"
6. Only provide information you have access to
7. Do not provide any further explanations
""",
        ),
        ("user", "{user_question}"),
    ]
)

explanation_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert medical educator providing explanations for medical exam questions.

Patient Information Available:
- Question: {question}
- Clinical Context: {context} 
- Answer Choices: {options}
- Correct Answer: {answer}
- Student's Answer: {userAnswer}
- Lab Results: {labResults}
- Blood Pressure: {bloodPresure}
- Pulse: {pulse}
- Respirations: {respirations}
- Physical Examination: {physicalExamination}
- Temperature: {temperature}
- History: {history}
- Demographics: {demographics}
- Family History: {familyHistory}
- Allergies: {allergies}
- History: {history}
- Medications: {medications}

Your Teaching Approach:
1. **Confirm the correct answer** and explain why it's right
2. **Address incorrect choices** - explain why each wrong option doesn't fit
3. **If student answered incorrectly:**
   - Analyze their reasoning process
   - Identify what clinical clues they may have missed
   - Guide them to the correct diagnostic pathway

Response Guidelines:
- **Be concise by default** - keep explanations brief and focused
- **For simple questions** (what, when, basic facts): Answer in 1-2 sentences
- **For complex questions** (why, how, pathophysiology): Provide detailed explanations when necessary
- **Use clear medical reasoning** with supporting evidence
- **Maintain an encouraging, educational tone**
- **For non-medical questions**: Respond with "I'm not ChatGPT. Put some respect on my name."

Example Concise Response:
"The correct answer is C (17 alpha-hydroxylase deficiency). This enzyme deficiency causes lack of sex hormone production while maintaining mineralocorticoid activity, explaining the patient's absent puberty, hypertension, and 46,XY karyotype with female phenotype."
""",
        ),
        ("user", "{user_question}"),
    ]
)


class State(MessagesState):
    """Extended MessagesState with medical case information."""

    context: str
    question: str
    answer: str
    userAnswer: str
    questionAnswered: bool

    # Optional medical information
    medications: list[str]
    allergies: list[str]
    familyHistory: list[str]
    labResults: list[str]
    options: list[str]
    bloodPresure: str
    respirations: str
    pulse: str
    physicalExamination: str
    temperature: str
    history: str
    demographics: str


graph_builder = StateGraph(State)


def get_response(state: State) -> State:
    """Determine which prompt to use and get a response from the LLM.

    If the user has not answered the question, you will not provide the answer or any explanation of the answer choices.
    You can only provide the details of the patients and the context provided to you.
    Once the user has answered the question, you can provide clarifications of the answer choices.
    """
    user_question = state["messages"][-1]
    if state["questionAnswered"]:
        prompt = explanation_prompt_template.invoke(
            {**state, "user_question": user_question.content}
        )
    else:
        prompt = context_prompt_template.invoke(
            {**state, "user_question": user_question.content}
        )

    response = llm.invoke(prompt)
    return {
        "messages": [response],
        "questionAnswered": state["questionAnswered"],
        "userAnswer": state["userAnswer"],
        "context": state["context"],
        "question": state["question"],
        "answer": state["answer"],
        "options": state["options"],
        "labResults": state["labResults"],
        "bloodPresure": state["bloodPresure"],
        "respirations": state["respirations"],
        "pulse": state["pulse"],
        "physicalExamination": state["physicalExamination"],
        "temperature": state["temperature"],
        "history": state["history"],
        "demographics": state["demographics"],
        "familyHistory": state["familyHistory"],
        "allergies": state["allergies"],
        "medications": state["medications"],
    }


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("llm_responder", get_response)

graph_builder.add_edge(START, "llm_responder")
graph_builder.add_edge("llm_responder", "__end__")

graph = graph_builder.compile()
