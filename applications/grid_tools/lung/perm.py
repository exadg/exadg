import itertools

As = list(itertools.permutations(range(0,4)))
Bs = list(itertools.permutations(range(4,8)))

for a in As:
	for b in Bs:
		print " ".join([ str(e) for e in a])+" "+" ".join([str(e) for e in  b])

