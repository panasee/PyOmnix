from langchain.tools import Tool
from langchain_google_commmunity import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper(k=7)

google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results",
    func=search.run,
)