import sys
import json
from yachalk import chalk
from groq import Groq
import os

sys.path.append("..")

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
llm_client = Groq(api_key=api_key)

def extractConcepts(prompt: str, metadata={}, model="mixtral-8x7b-32768"):

    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_concepts",
                "description": "Extract the most important and atomistic concepts from the text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "extracted_concepts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {
                                        "type": "string",
                                        "description": "The Concept or entity extracted."
                                    },
                                    "importance": {
                                        "type": "integer",
                                        "description": "The contextual importance of the concept on a scale of 1 to 5 (5 being highest)."
                                    },
                                    "category": {
                                        "type": "string",
                                        "enum": ["event", "concept", "place", "object", "document", "organisation", "condition", "misc"],
                                        "description": "The type of concept."
                                    }
                                },
                                "required": ["entity", "importance", "category"]
                            }
                        }
                    },
                    "required": ["extracted_concepts"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "Your task is to extract the key concepts (and non-personal entities) mentioned in the given context. Extract only the most important and atomistic concepts, breaking them down into simpler concepts if needed."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        # Standard Groq API call for tool use
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "record_concepts"}}
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            arguments = json.loads(tool_calls[0].function.arguments)
            result = arguments.get("extracted_concepts", [])
            
            # Inject metadata
            result = [dict(item, **metadata) for item in result]
            return result
        else:
            print("\n\nERROR ### No tool calls returned.\n\n")
            return None

    except Exception as e:
        print(f"\n\nERROR ### Failed during Groq generation or parsing: {e}\n\n")
        return None


def graphPrompt(input: str, metadata={}, model="mixtral-8x7b-32768"):

    tools = [
        {
            "type": "function",
            "function": {
                "name": "record_graph_edges",
                "description": "Extract an ontology of terms and the relationships between them based on the context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "node_1": {
                                        "type": "string",
                                        "description": "A concept from the extracted ontology."
                                    },
                                    "node_2": {
                                        "type": "string",
                                        "description": "A related concept from the extracted ontology."
                                    },
                                    "edge": {
                                        "type": "string",
                                        "description": "The relationship between the two concepts, node_1 and node_2, in one or two sentences."
                                    }
                                },
                                "required": ["node_1", "node_2", "edge"]
                            }
                        }
                    },
                    "required": ["edges"]
                }
            }
        }
    ]

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context.\n"
        "Thought 1: While traversing through each sentence, think about the key terms mentioned in it.\n"
        "Terms may include object, entity, location, organization, person, condition, acronym, documents, service, concept, etc.\n"
        "Terms should be as atomistic as possible.\n\n"
        "Thought 2: Think about how these terms can have a one-on-one relation with other terms.\n"
        "Terms mentioned in the same sentence or paragraph are typically related.\n"
        "Terms can be related to many other terms.\n\n"
        "Thought 3: Find out the relation between each such related pair of terms and record them using the provided tool."
    )

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"context: ```{input}```"}
    ]

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "record_graph_edges"}}
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            arguments = json.loads(tool_calls[0].function.arguments)
            result = arguments.get("edges", [])
            
            # Inject metadata
            result = [dict(item, **metadata) for item in result]
            return result
        else:
            print("\n\nERROR ### No tool calls returned.\n\n")
            return None

    except Exception as e:
        print(f"\n\nERROR ### Failed during Groq generation or parsing: {e}\n\n")
        return None