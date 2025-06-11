from agents.loader import load_all_agents
growth = load_all_agents()["growth"]
idea = {"name": "HIPAA SaaS", "tag": "Comply in one click"}

tweets = growth.run(f"Generate Twitter thread for {idea['name']}")
print("\n".join(tweets))

journey = growth.run(f"Welcome-email flow for {idea['name']}")
print(journey)
