inference.py
	--data [file name of data in datasets/]
	--run_llm (toggle to run LLM - must have OPENAI_API_KEY set in env)

Folders:
	- datasets/:	Where the data will be looked for by inference.py
	- results/: 	Where outputs of the fake_news_classifer.ipynb and inference.py are saved
	- models/:	Where the Logistic Regression .pkl is saved - path is hardcoded into inference.py