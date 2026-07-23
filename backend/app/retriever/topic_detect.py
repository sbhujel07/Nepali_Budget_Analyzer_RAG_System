def detect_topic(user_query,topic_map):
    user_query = user_query.lower()

    for topic in topic_map.keys():
        if topic.lower() in user_query:
            return topic

    return "other"