import texttable as tt
import hmm_tags
HEADER = hmm_tags.WHITE_LIST

def disp_emm(model):
    data = [HEADER]
    x = tt.Texttable()
    x.add_row(HEADER)
    for row in model._hmm.emissionprob_:
        x.add_row(row)
    print(x.draw())