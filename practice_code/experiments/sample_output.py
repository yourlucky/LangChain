#Basic Code
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant that is role playing as a teacher.
        
Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

Each question should have 4 answers, three of them must be incorrect and one should be correct.
        
Use (o) to signal the correct answer.
        
Question examples:
        
Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)
        
Question: What is the capital or Georgia?
Answers: Baku|Tbilisi(o)|Manila|Beirut
                
Your turn!
        
Context: {context}
""",
        )
    ]
)

chain = {"context": format_docs} | prompt | llm