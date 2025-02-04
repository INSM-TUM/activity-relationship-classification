import textwrap

base_text = """You are an assistant business process re-designer. Your job is to explain the context behind the ordering of a pair of activities, given the pair of activities and the process description, by categorizing the reason of the specific order in zero or one of the three following categories:
1- Governmental Law: Rules created and enforced by governmental institutions to regulate business behaviour (i.e. Customer cannot cash a cheque without validating their documents).
2- Best Practice: Procedures usually accepted by the organization's staff or industry-wide to be superior to alternatives, but are not required to be followed nor enforced by any stakeholder (i.e. Following up with patients after treatment).
3- Business Rule: Rules that are under full jurisdiction of the stakeholders of the process (i.e. organization or suppliers) that can change or discard this rule at their own discretion (i.e. holding regular meetings after starting project).

Separate from the categories, you need to decide if the relationship is due to a law of nature, which is an inviolable relationship where the second activity cannot precede the first activity due to either a deadlock occurring or due to a data (i.e. You cannot reply to a message without receiving it), resource dependency (i.e. You cannot print a document without having paper), or logical dependency from the first activity."""


how_to_format = """Structure your answer in the following format without any additional text, and replace the placeholders with the correct values:
{
    "First Activity": "-",
    "Second Activity": "-",
    "Category": "-",
    "Justification": "-",
    "Law of Nature": "-"
}

If none of the three contextual origin categories apply to the relationship, put a dash in both the Category and Justification fields and do not include any other text for justifying your decision. Otherwise, put the category you chose in the "Category" field, the justification for your choice in the "Justification" field. In the "Law of Nature" field, if the answer is yes, then you should put justification in the value, if it is no, then only put a single dash."""

what_queries = """You will receive the prompt as "which of the categories best describes the contextual origin of why [First Activity]  occurs before [Second Activity]?", the first activity always occurs in time before the second activity. Return only the JSON response with no other text outside the JSON."""

cot_queries = "Before you make a decision, you should think logically about the classification task. Work it out in a step-by-step way to be sure to have the right answer."

def create_query(activity1, activity2, context):
    return f"Based on the following context, decide which of the categories best describes the contextual origin of why {activity1} occurs before {activity2}? Explain why you chose this category and not another one. If you think none of the categories fit the pair of activities, you do not have to choose a category. After discussing the contextual origin, discuss if the ordering is due to a law of nature.\n\nContext:\n{context}"




