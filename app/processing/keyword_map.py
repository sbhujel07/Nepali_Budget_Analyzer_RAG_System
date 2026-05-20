keyword_map = {
    "शिक्षा": ["विद्यालय", "शिक्षा", "विद्यार्थी", "पाठ्यक्रम", "शिक्षण"],
    
    "स्वास्थ्य": ["स्वास्थ्य", "चिकित्सा", "उपचार", "डाक्टर", "अस्पताल", "औषधि"],
    
    "अर्थतन्त्र": [
        "अर्थतन्त्र", "अर्थव्यवस्था", "आर्थिक", "बजेट","राजस्व", "आय", "वित्त", "लगानी", "सुधार"
         ],
    
    "कृषि": ["कृषि", "खेती", "बाली", "उत्पादन", "किसान"],
    
    "पूर्वाधार": ["सडक", "पूर्वाधार", "पुल", "सञ्चार", "पानी", "विद्युत"],
    
    "विज्ञान तथा प्रविधि": ["विज्ञान", "प्रविधि", "आईटी", "डिजिटल", "अनुसन्धान"],
    
    "उद्योग र व्यापार": ["उद्योग", "व्यापार", "उद्यम", "कारखाना", "नौकरी"],
    
    "पर्यटन": ["पर्यटन", "यात्रा", "टुरिज्म"],
    
    "सामाजिक सुरक्षा": ["पेंशन", "सामाजिक सुरक्षा", "अनुदान", "भत्ता"]
}

#return score for keyword mapping
def keyword_matching(sentence, topic, keyword_map):
    """Count how many keywords from a specific topic appear in the sentence"""
    score = 0
    #check if topic is in keyword map 
    if topic in keyword_map:
        #calc the values of topic
        for keyword in keyword_map[topic]:
            #check if values in sentence
            if keyword in sentence:
                score += 1
    return score