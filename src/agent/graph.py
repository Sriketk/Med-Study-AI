from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

# - Temperature: {temperature}
# - History: {history}
# - Demographics: {demographics}
# - Family History: {familyHistory}
# - Allergies: {allergies}
# - History: {history}
# - Medications: {medications}





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
            """You are an expert medical professional providing detailed explanations for medical exam questions.

Given:
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


Your role is to:
1. Confirm the correct answer and provide a comprehensive explanation of why it is correct
2. Explain why each incorrect answer choice is wrong
3. If the student answered incorrectly:
   - Analyze their likely reasoning that led to that choice
   - Identify specific elements in the question stem or clinical context that, if different, would have made their answer correct
   - Guide them toward understanding why the correct answer is more appropriate

Focus on being clear, thorough, and educational in your explanations while maintaining a supportive tone.

Example Response:
"Let me explain why option B (Beta-blockers) is incorrect for this patient's hypertension management.

The key reasons why beta-blockers would not be the optimal choice:

1. Patient Characteristics:
- The patient is a 68-year-old African American female
- Has uncomplicated essential hypertension
- Blood pressure reading of 152/91 mmHg

2. Why Beta-blockers are not ideal:
- Beta-blockers are no longer recommended as first-line therapy for uncomplicated hypertension in elderly patients
- African American patients typically respond less favorably to beta-blockers compared to other antihypertensive medications
- They can mask symptoms of hypoglycemia in diabetic patients
- May cause fatigue and decreased exercise tolerance in elderly patients

3. Better alternatives would include:
- Thiazide diuretics
- Calcium channel blockers
These medications have shown better efficacy in African American patients and elderly populations.

4. When beta-blockers would be appropriate:
- If the patient had concurrent conditions like:
  * Coronary artery disease
  * Heart failure
  * Tachyarrhythmias
  * Post-myocardial infarction

Understanding the patient's demographics and comorbidities is crucial in selecting the most appropriate antihypertensive medication. The choice should align with current guidelines while considering individual patient factors."
""",
        ),
        ("user", "{user_question}"),
    ]
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    question: str
    answer: str
    userAnswer: str | None = ""
    questionAnswered: bool = False
     
    # demographics: list[str] | None = []
    # history: list[str] | None = []
    medications: list[str] | None = []
    allergies: list[str] | None = []
    familyHistory: list[str] | None = []
    labResults: list[str] | None = []
    options: list[str] | None = []
    # temperature: str | None = ""
    bloodPresure: str | None = ""
    respirations: str | None = ""
    pulse: str | None = ""
    physicalExamination: str | None = ""
    


graph_builder = StateGraph(State)


def get_response(state: State):
    """Determine which prompt to use and get a response from the LLM."""
    user_question = state["messages"][-1]
    if state["questionAnswered"]:
        prompt = explanation_prompt_template.invoke({**state, "user_question": user_question.content})
    else:
        prompt = context_prompt_template.invoke({**state, "user_question": user_question.content})

    response = llm.invoke(prompt)
    return {"messages": [response]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("llm_responder", get_response)

graph_builder.add_edge(START, "llm_responder")
graph_builder.add_edge("llm_responder", "__end__")
graph = graph_builder.compile()
