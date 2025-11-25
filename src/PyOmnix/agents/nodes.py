from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pyomnix.agents.states import ConversationState

def call_model(model, state: ConversationState, config: RunnableConfig):
    """
    general chat node: support chat with summary of previous conversation
    Args:
        model: BaseChatModel | _ConfigurableModel
        state: ConversationState
        config: RunnableConfig
    Returns:
        dict[str, Any]
    """
    
    summary = state.get("summary", "")
    
    if summary:
        system_message = f"Previous conversation summary: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
        
    response = model.invoke(messages)
    return {"messages": [response]}

def summarize_conversation(model, state: ConversationState, config: RunnableConfig):
    """
    Summarize the conversation.
    Args:
        model: BaseChatModel | _ConfigurableModel
        state: ConversationState
        config: RunnableConfig
    Returns:
        dict[str, Any]
    """
    summary = state.get("summary", "")
    messages = state["messages"]
    if summary:
        summary_message = (
            f"Current conversation summary: {summary}\n\n"
            "Please merge the summary with the new conversation content and update it to a new summary."
            "【Important instructions】Please keep the summary language consistent with the main language of the conversation content."
            "If the conversation is in English, use English summary, if the conversation is in Chinese, use Chinese summary, if the conversation is mixed, judge it yourself."
        )
    else:
        summary_message = (
            "Please summarize the following conversation content into a concise paragraph."
            "【Important instructions】Please keep the summary language consistent with the main language of the conversation content."
        )
    messages_to_summarize = messages[:-2] # keep the last 2 messages
    if not messages_to_summarize:
            return {}

    prompt = ChatPromptTemplate.from_messages([
        ("system", summary_message),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Start summarizing now.")
    ])
    
    chain = prompt | model
    response = chain.invoke({"history": messages_to_summarize})
    new_summary = response.content

    # remove old messages
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]

    return {
        "summary": new_summary, 
        "messages": delete_messages 
    }