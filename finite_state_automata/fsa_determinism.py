class FSA:
    def __init__(self, num_states = 0):
        self.num_states = num_states
        self.transitions = {}
        self.final_states = set()
        self.symbols = {}

    # add transitions
    def add_trasition (self, from_state, to_state, symbol):
        assert from_state < self.num_states
        assert to_state < self.num_states
        self.symbols[symbol] = True
        self.transitions[(from_state, symbol)] = to_state

    # set final states 
    def set_final (self, the_state):
        assert the_state < self.num_states
        if not the_state in self.final_states:
            self.final_states.add(the_state)

    # look up next state in the state transitions table
    def next_state (self, from_state, symbol):
        if (from_state,symbol) in self.transitions.keys():
            return self.transitions[(from_state, symbol)]
        else:
            return ""

    # check whether or not a state is a final (accepting) state
    def is_final (self, the_state):
        if the_state in self.final_states:
            return True
        else:
            return False

    # print the FSA (debug use only) 
    def print_fsa(self):
        for from_state in range (self.num_states):
            for symbol in self.symbols.keys():
                if (from_state,symbol) in self.transitions.keys():
                    to_state = self.transitions[(from_state, symbol)]
                    print("\t".join([str(from_state), symbol, str(to_state)]))
            if from_state in self.final_states:
                print(from_state)
# months
months = FSA(4)
months.set_final(3)
months.add_trasition(0, 1, "0")
months.add_trasition(0, 2, "1")
for i in range (1, 10):
    months.add_trasition(1, 3, str(i))
months.add_trasition(2, 3, "0")
months.add_trasition(2, 3, "1")
months.add_trasition(2, 3, "2")

# days
days = FSA(5)
days.set_final(4)
days.add_trasition(0, 1, "0")
days.add_trasition(0, 2, "1")
days.add_trasition(0, 2, "2")
days.add_trasition(0, 3, "3")
for i in range (1, 10):
    days.add_trasition(1, 4, str(i))
for i in range (0, 10):
    days.add_trasition(2, 4, str(i))
days.add_trasition(3, 4, "0")
days.add_trasition(3, 4, "1")

#years
years = FSA(6)
years.set_final(5)
years.add_trasition(0, 1, "1")
years.add_trasition(0, 2, "2")
years.add_trasition(1, 3, "9")
years.add_trasition(2, 3, "0")
for i in range (0, 10):
    years.add_trasition(3, 4, str(i))
for i in range (0, 10):
    years.add_trasition(4, 5, str(i))

#separators
seps = FSA(2)
seps.set_final(1)
seps.add_trasition(0, 1, " ")
seps.add_trasition(0, 1, "-")
seps.add_trasition(0, 1, "/")

# DRecognize returns True if the fsa accepts the input; return false otherwise.
def DRecognize(input, fsa):
    index = 0
    cur_state = 0
    while (True):
        if index == len(input):
            if fsa.is_final(cur_state):
                return True
            else:
                return False
        elif fsa.next_state(cur_state, input[index]) == " ":
            return False
        else:
            cur_state = fsa.next_state(cur_state, input[index])
            index = index + 1


# DRecognizeMulti returns True if the concatenation fo the fsa_list accepts the input;
# return false otherwise.
def DRecognizeMulti(input, fsa_list):
    input_index = 0
    cur_state = 0
    fsa_index = 0
    while (True):
        if input_index == len(input):
            if fsa_list[fsa_index].is_final(cur_state) and fsa_index == len(fsa_list) - 1:
                return True
            else:
                return False

        if fsa_list[fsa_index].is_final(cur_state):
            cur_state = 0
            fsa_index = fsa_index + 1

        if fsa_index >= len(fsa_list):
            return False

        if fsa_list[fsa_index].next_state(cur_state, input[input_index]) == " ":
            return False
        else:
            cur_state = fsa_list[fsa_index].next_state(cur_state, input[input_index])
            input_index = input_index + 1
    

# test
def Test(months, days, years, seps):
    print "\nTest Months FSA"
    for input in ["", "00", "01", "09", "10", "11", "12", "13", "100"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [months]))
    print "\nTest Days FSA"
    for input in ["", "00", "01", "09", "10", "11", "21", "31", "32", "100"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [days]))
    print "\nTest Years FSA"
    for input in ["", "1899", "1900", "1901", "1999", "2000", "2001", "2099", "2100", "190", "20030"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [years]))
    print "\nTest Separators FSA"
    for input in ["", ",", " ", "-", "/", "//", ":"]:
        print "'%s'\t%s" %(input, DRecognizeMulti(input, [seps]))
    print "\nTest Date Expressions FSA"
    for input in ["", "12 31 2000", "12/31/2000", "12-31-2000", "12:31:2000", 
                  "00-31-2000", "12-00-2000", "12-31-0000", 
                  "12-32-1987", "13-31-1987", "12-31-2150", "12-31-19999"]:
        print "'%s'\t%s" %(input, 
                           DRecognizeMulti(input, [months, seps, days, seps, years]))

Test(months, days, years, seps)