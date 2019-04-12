import itertools

As = list(itertools.permutations(range(0,4)))
Bs = list(itertools.permutations(range(4,8)))

for a in reversed(As):
	for b in reversed(Bs):
		print " ".join([ str(e) for e in a])+" "+" ".join([str(e) for e in  b])

