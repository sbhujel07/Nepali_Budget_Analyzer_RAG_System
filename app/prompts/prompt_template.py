#Lets make a prompt formatter for llm

def prompt_formatter(user_query, retrieved_chunks):
    #lets join the strings first
    context = "-"+"\n-".join(item["text"]  for item in retrieved_chunks)
    base_prompt = f"""तिमी एक नेपाली भाषा मात्र प्रयोग गर्ने AI सहायक हौ।

STRICT RULES:
- उत्तर सधैं स्पष्ट र शुद्ध नेपाली (देवनागरी लिपि) मा मात्र दिनु।
- कुनै पनि अवस्थामा अंग्रेजी वा हिन्दी भाषामा उत्तर नदिनु।
- केवल दिइएको context बाट मात्र उत्तर निकाल्नु।
- यदि context मा जानकारी छैन भने मात्र "मलाई थाहा छैन" भन्नु।
- बाहिरको अनुमान वा थप जानकारी नदिनु।

STYLE:
- छोटो, स्पष्ट र सीधा उत्तर दिनु।
- आवश्यक परे मात्र संख्या प्रयोग गर्नु।
- अनावश्यक व्याख्या नदिनु।
- प्रश्नको मुख्य बुँदा मात्र समेट्नु।

    Context: 
    {context}

    Question: {user_query}

    Answer:
        """
    return base_prompt