from ner import Parser

p = Parser()

p.load_models("models/")

print(p.predict("Sarah works in Amazon"))

