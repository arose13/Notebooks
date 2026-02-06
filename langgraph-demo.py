"""
The required dependencies for this demo are:
- langgraph
- langgraph-cli[inmem]
- langchain

then use `langgraph dev` to run in the browser!
"""
import time
import random

from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

delay_range = (0.5, 1.0)

class DemoState(TypedDict):
    message: str

#################################################################################################
# Nodes
#################################################################################################
def node_a(state: DemoState) -> DemoState:
    time.sleep(random.uniform(*delay_range))
    state['message'] += ' -> A'
    return state

def node_b(state: DemoState) -> DemoState:
    time.sleep(random.uniform(*delay_range))
    state['message'] += ' -> B'
    return state

def node_c(state: DemoState) -> DemoState:
    time.sleep(random.uniform(*delay_range))
    state['message'] += ' -> C'
    return state

def route_c(state: DemoState) -> str:
    return random.choice(['A', 'B', 'C', END])


#################################################################################################
# Graph
#################################################################################################
graph = StateGraph(DemoState)
graph.add_node('A', node_a)
graph.add_node('B', node_b)
graph.add_node('C', node_c)

graph.add_edge(START, 'A')
graph.add_edge('A', 'B')
graph.add_edge('B', 'C')
graph.add_conditional_edges('C', route_c, ['A', 'B', 'C', END])

app = graph.compile()