import tiktoken

print(tiktoken)

examples = [
            {"question": "What do you know about France?", 
            "answer": """
                Here is what I know: ",
                Capital: Paris ",
                Language: French ",
                Food: Wine and Cheese ",
                Currency: Euro ",
                """,
            },
            {
            "question": "What do you know about Italy?",
            "answer": """
                I know this: ",
                Capital: Rome ",
                Language: Italian ",
                Food: Pizza and Pasta ",
                Currency: Euro ",
                """,
            },
            { 
            "question": "What do you know about Greece?",
                "answer": """
                I know this: ",
                Capital: Athens ",
                Language: Greek ",
                Food: Souvlaki and Feta Cheese ",
                Currency: Euro ",
                """,
            },
]