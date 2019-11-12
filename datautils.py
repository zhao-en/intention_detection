import numpy as np
from sklearn.preprocessing import LabelEncoder

out = ['我','是','中','国','人','我','是','是','好','人']
outy = LabelEncoder().fit_transform(out)
# out.sort(reverse= True);

# c = dict(list(zip(out,np.arange(len(out)))))

debug = 0;