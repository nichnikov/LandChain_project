from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-or-vv-17bd53f8f505e0a1d24a3bf0a8bb702e13edbc78c89ac9aa41f6bda7ec72270c",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="https://api.vsegpt.ru/v1",
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)

sudoku_puzzle =   "3,*,*,2|1,*,*,4|2,*,*,3|*,3,*,1"
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
problem_description = f"""
{sudoku_puzzle}

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()
print(problem_description)

#######
# The following code implement a simple rule based checker for 
# a specific 4x4 sudoku puzzle.
#######

from typing import Tuple
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
import re

class MyChecker(ToTChecker):
    def evaluate(self, problem_description: str, thoughts: Tuple[str, ...] = ()) -> ThoughtValidity:
        last_thought = thoughts[-1]
        print(f"Last thought: {last_thought}")
        clean_solution = last_thought.replace(" ", "").replace('"', "")
        print(f"Clean solution: {clean_solution}")
        regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
        print(f"Regex solution: {regex_solution}")
        if sudoku_solution in clean_solution:
            return ThoughtValidity.VALID_FINAL
        elif re.search(regex_solution, sudoku_solution):
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID

#######
# Testing the MyChecker class above:
#######

checker = MyChecker()
assert checker.evaluate("", ("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1",)) == ThoughtValidity.VALID_INTERMEDIATE
assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",)) == ThoughtValidity.VALID_FINAL
assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",)) == ThoughtValidity.VALID_INTERMEDIATE
assert checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,*,3,1",)) == ThoughtValidity.INVALID

#######
# Initialize and run the ToT chain, 
# with maximum number of interactions k set to 30 and 
# the maximum number child thoughts c set to 8.
#######

from langchain_experimental.tot.base import ToTChain

tot_chain = ToTChain(llm=llm, checker=MyChecker(), k=30, c=5, verbose=True, verbose_llm=False)
tot_chain.run(problem_description=problem_description)