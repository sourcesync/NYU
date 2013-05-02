class LanguageModel:
    def __init__(self):
        self.prefixes = {}

        for line in file("generaldict1"):
            word, freq = line.split()
            word = word + " "
            freq = int(freq)

            for i in range(len(word)):
                prefix = word[:i+1]
                self.prefixes[prefix] = self.prefixes.get(prefix, 0) + freq

            
#         if words is None:
#             words = wordsfromfile("/home/pz215/myfiles/dasher/dasher3/Data/system.rc/training_english_GB.txt")

#         for word in words:
#             word = word + " "
#             for i in range(len(word)):
#                 prefix = word[:i+1]
#                 self.prefixes[prefix] = self.prefixes.get(prefix, 0) + 1



    def getprobs(self, prefix, names):
        probs = [self.prefixes.get(prefix + name, 0) for name in names]
        total = float(sum(probs)) + 1
        return [prob / total + 0.01 for prob in probs]

def wordsfromfile(filename):
    for line in file(filename):
        for word in line.split():
            yield word.lower()
