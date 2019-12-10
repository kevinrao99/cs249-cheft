import heapq
import math
import matplotlib.pyplot as plt
import random
from karp import Graph # Credits go to Wikipedia

# Homogeneous-Abstracted Scheduling with Critical Heuristic

n_tasks = 0

succ = [] # adjacency list storing all successors of a vertex, 1 indexed
pred = [] # same but predecessors
weights = []
EST = [] # earliest start time
V_ENTRY = 0
V_EXIT = -1
LST = [] # latest start time
CNP = [] # true if task is a critical node predecessor and is not a critical node itself
CNCT = [] # Fuck a beat, I was tryna beat a case
PCT = [] # max value of longest path from successors of node to V_EXIT
ROUND = 4 # digits past decimal to round to
max_antichain = 0
max_weight_path = 0

topo = []
rev_topo = []


def compute_max_antichain():
	global max_antichain

	bigraph = [[0 for i in range(2 * n_tasks + 2)] for j in range(2 * n_tasks + 2)]
	for i in range(0, n_tasks):
		bigraph[2 * n_tasks][i] = 1

	for i in range(n_tasks, 2 * n_tasks):
		bigraph[i][2 * n_tasks + 1] = 1

	for i in range(len(succ)):
		for v in succ[i]:
			bigraph[i][n_tasks + v] = 1

	bigraph_obj = Graph(bigraph)

	max_antichain = n_tasks - bigraph_obj.edmonds_karp(2 * n_tasks, 2 * n_tasks + 1)

def compute_toposort():
	global topo
	global rev_topo
	visited = [False for i in range(n_tasks)]
	topo = []
	rev_topo = []

	def visit(v):
		if visited[v]:
			return
		visited[v] = True
		for child in succ[v]:
			visit(child)
		rev_topo.append(v)
	visit(0)
	for i in range(len(rev_topo)):
		topo.append(rev_topo[len(rev_topo) - i - 1])

def compute_max_weight_path():
	global max_weight_path
	dis_to_end = [0 for i in range(n_tasks)]
	for v in rev_topo:
		l_max = 0
		for v_succ in succ[v]:
			l_max = max(l_max, dis_to_end[v_succ])
		dis_to_end[v] = weights[v] + l_max
	max_weight_path = dis_to_end[0]

def compute_EST():
	global EST
	EST = []

	for i in range(n_tasks):
		if i == 0:
			EST.append(0)
		else:
			EST.append(-1)

	for v in topo:
		l_max = 0
		for p in pred[v]:
			l_max = max(EST[p] + weights[p], l_max)
		EST[v] = l_max

def compute_LST():
	global LST
	LST = []

	for i in range(n_tasks):
		if i == 0:
			LST.append(0)
		else:
			LST.append(-1)

	for v in rev_topo:
		if len(succ[v]) == 0:
			LST[v] = EST[v]
		else:
			l_min = LST[succ[v][0]]
			for s in succ[v]:
				l_min = min(LST[s], l_min)
			LST[v] = l_min - weights[v]

def compute_CNP():
	global CNP
	CNP = [False for i in range(n_tasks)]
	for v in range(n_tasks):
		res = False
		for s in succ[v]:
			res = res or EST[s] == LST[s]
		CNP[v] = res and not EST[v] == LST[v]

def compute_CNCT(): # error test this shit
	global CNCT
	CNCT = [0 for i in range(n_tasks)]
	for v in rev_topo:
		has_CN = False
		for s in succ[v]:
			has_CN = has_CN or EST[s] == LST[s]
		l_max = 0
		for s in succ[v]:
			if not has_CN or (has_CN and EST[s] == LST[s]): # only consider this successor in the max if there are no critical node successors or there are and this is one of them
				l_max = max(l_max, CNCT[s] + weights[s])
		CNCT[v] = l_max

def compute_PCT():
	global PCT
	PCT = [0 for i in range(n_tasks)]
	for v in rev_topo:
		l_max = 0
		for v_succ in succ[v]:
			l_max = max(l_max, PCT[v_succ] + weights[v_succ])
		PCT[v] = l_max

def init_from_file(): # Read & initialize n_tasks, V_EXIT, succ, pred, weights from file
	global n_tasks
	global succ
	global pred
	global weights
	global V_EXIT
	f = open("input_hasch.in", "r")
	[a] = list(map(int, f.readline().rstrip().split()))
	n_tasks = a
	V_EXIT = n_tasks - 1

	#file is succ first then pred, each n_tasks lines


	for i in range(n_tasks):
		succ.append(list(map(int, f.readline().rstrip().split())))

	for i in range(n_tasks):
		pred.append(list(map(int, f.readline().rstrip().split())))


	weights = list(map(int, f.readline().rstrip().split()))

	# we make the simplifying assumption for now that the communication cost is always 0

def binomial_init_graph(n, p): # n is number of nontrivial nodes, p is probability that any edge is in the graph
	# notably, may generate a graph that is not connected
	# notably does not init weights
	global n_tasks
	global succ
	global pred
	global V_EXIT
	n_tasks = n + 2
	V_EXIT = n_tasks - 1
	succ = []
	pred = []

	for i in range(n_tasks):
		pred.append([])
		succ.append([])

	for i in range(n_tasks):
		if i == 0:
			for j in range(1, V_EXIT):
				succ[i].append(j)
				pred[j].append(i)
		elif i != V_EXIT:
			for j in range(i + 1, V_EXIT):
				if random.random() < p:
					succ[i].append(j)
					pred[j].append(i)
		else:
			for j in range(0, V_EXIT):
				pred[i].append(j)
				succ[j].append(i)

def unif_init_weights(left, right):
	global weights
	weights = [0 for i in range(n_tasks)]
	for i in range(len(weights)):
		weights[i] = left + random.random() * (right - left)
	weights[V_ENTRY] = 0
	weights[V_EXIT] = 0

def graph_preprocess(): # Run all preprocessing functions
	compute_toposort()
	compute_EST()
	compute_LST()
	compute_CNP()
	compute_CNCT() # try to come up with a test case where CNCT is not the same as PCT
	compute_PCT() 
	compute_max_antichain()
	compute_max_weight_path()

def HASCH(processors): # referred to as CHEFT in the paper
	processor_schedule = []
	n_processors = len(processors)

	for i in range(n_processors):
		processor_schedule.append([]) # array of arrays, one per process. A task is a triple (start, end, task) in the array

	AFT = [-1 for i in range(n_tasks)]

	def time_available(task, p): # finds earliest time we can process task on processor p
		time_needed = round(float(weights[task]) / processors[p], ROUND)
		last_ended = 0
		for task_pred in pred[task]:
			last_ended = max(last_ended, AFT[task_pred])
		for i in range(len(processor_schedule[p])):
			(start, end, t) = processor_schedule[p][i]
			if start - last_ended >= time_needed:
				return last_ended
			last_ended = max(last_ended, end)
		return round(last_ended, ROUND)

	def compute_EFT(v, p):
		T_available = time_available(v, p)
		l_max = 0
		for v_pred in pred[v]:
		#	print AFT, v_pred, v
			assert AFT[v_pred] >= 0
			l_max = max(l_max, AFT[v_pred])
		return max(l_max, T_available) + round(float(weights[v]) / processors[p], ROUND)

	def assign(task, p): # assigns the task to the processor, earliest processing time
		if task == V_ENTRY:
			AFT[task] = 0
			return
		insert_time = time_available(task, p)
		insert_index = 0

		for item in processor_schedule[p]:
			(start, end, _) = item
			if start <= insert_time:
				insert_index += 1

		temp_aft = insert_time + round(float(weights[task]) / processors[p], ROUND)
		processor_schedule[p].insert(insert_index, (insert_time, round(temp_aft, ROUND), task))
		AFT[task] = temp_aft

	task_order = []
	for i in range(len(PCT)):
		heapq.heappush(task_order, (-1 * PCT[i], -1 * weights[i], i)) # Rank by PCT, break ties with weight

	# v print PCT
	while len(task_order) > 0:
		(_, _, v) = heapq.heappop(task_order)
		EFT_CNCT = [0 for i in range(n_processors)]
		for p in range(n_processors):
			EFT = compute_EFT(v, p)
			if not CNP[v]:
				EFT += CNCT[v]
			EFT_CNCT[p] = EFT

		min_eft_cnct = EFT_CNCT[0]
		min_processor = 0
		for i in range(len(EFT_CNCT)):
			if min_eft_cnct >= EFT_CNCT[i]:
				if min_eft_cnct > EFT_CNCT[i] or processors[i] > processors[min_processor]:
					min_eft_cnct = EFT_CNCT[i]
					min_processor = i

		assign(v, min_processor)

	return (AFT[V_EXIT], processor_schedule)

def baseline(processors): # Theoretical optimal scheduling time, based on minimum path cover

	tot_weight = 0
	for w in weights:
		tot_weight += w

	max_parallel_power = 0 # we assume an optimal scheduler has the fastest <avg_parallel> chips constantly running
	processors.sort(reverse = True)
	chips_left = max_antichain
	for p in processors:
		if chips_left < 0:
			break
		else:
			max_parallel_power += p * min(chips_left, 1)
			chips_left -= 1
	# print processors, max_parallel_power

	return float(tot_weight) / max_parallel_power

subplot_ct = 1

def run_baseline_v_HASCH_subsets(processor_speeds, plot_results = True): # array of (cost, speed) tuples
	global subplot_ct
	bin_rep = 1
	def f(processor_speed): # we use some function of processor speed as a proxy for the cost
		return processor_speed
		# return processor_speed ** 2
	def next_subset(seed):
		if seed >= 2**len(processor_speeds):
			return (-1, [])
		temp_seed = seed
		tot_cost = 0
		ans = []
		for i in range(len(processor_speeds)):
			if temp_seed % 2 == 1:
				speed = processor_speeds[i]
				ans.append(speed)
				tot_cost += f(speed)
			temp_seed /= 2
		return (tot_cost, ans)

	costs = []
	HASCH_times = []
	baseline_times = []

	avg_parallel = 0

	while(True):
		(subset_cost, processors) = next_subset(bin_rep)
		bin_rep += 1
		if subset_cost < 0:
			break
		costs.append(subset_cost)

		(finish_time, _) = HASCH(processors)
		HASCH_times.append(finish_time)

		finish_time = baseline(processors)
		baseline_times.append(finish_time)


	if plot_results:
		plt.subplot(1, 4, subplot_ct)
		subplot_ct += 1
		plt.scatter(costs, baseline_times, c = "Red", marker = '.', alpha = .5, label = "Baseline")
		plt.scatter(costs, HASCH_times, c = "Blue", marker = '.', alpha = .5, label = "CHEFT")
		plt.title("Max Parallelism: " + str(max_antichain))

def run_HASCH_subsets(processor_speeds, plot_results = True, targ_lat = None, hyp_speed = None):
	bin_rep = 1
	def f(processor_speed): # we use some function of processor speed as a proxy for the cost
		return processor_speed
		# return processor_speed ** 2
	def next_subset(seed):
		if seed >= 2**len(processor_speeds):
			return (-1, [])
		temp_seed = seed
		tot_cost = 0
		ans = []
		for i in range(len(processor_speeds)):
			if temp_seed % 2 == 1:
				speed = processor_speeds[i]
				ans.append(speed)
				tot_cost += f(speed)
			temp_seed /= 2
		return (tot_cost, ans)

	costs = []
	HASCH_times = []
	hyp_costs = []
	hyp_times = []

	while(True):
		(subset_cost, processors) = next_subset(bin_rep)
		bin_rep += 1
		if subset_cost < 0:
			break
		costs.append(subset_cost)

		(finish_time, _) = HASCH(processors)
		HASCH_times.append(finish_time)

	if not hyp_speed is None:
		bin_rep = 1
		while(True):
			(subset_cost, processors) = next_subset(bin_rep)
			if subset_cost < 0:
				break
			bin_rep += 1

			processors.append(hyp_speed)
			subset_cost += f(hyp_speed)
			hyp_costs.append(subset_cost)

			(finish_time, _) = HASCH(processors)
			hyp_times.append(finish_time)

	if plot_results:
		plt.scatter(costs, HASCH_times, c = "Blue", marker = '.', alpha = .5, label = "Existing Library")
		if not hyp_speed is None:
			plt.scatter(hyp_costs, hyp_times, c = "Green", marker = '.', alpha = .5, label = "With Hypothetical Processor")

		if not targ_lat is None:
			plt.ylim(0, targ_lat * 2.5)
			left, right = plt.xlim()
			plt.hlines(targ_lat, left, right, colors = "Red", label = "Target Latency")

		plt.xlabel("Cost of Processors")
		plt.ylabel("Latency")
		plt.title("MAVBench Package Delivery Dataflow")
		plt.legend()

		plt.show()

def SLR_comparison(processor_speeds, plot_results = True):

	processor_speeds.sort(reverse = True)

	SLR = []
	num_samples = 50
	num_tasks = []

	for i in range(10, 200, 10):
		acc = 0
		for j in range(num_samples):
			binomial_init_graph(i, .2)
			unif_init_weights(20, 150)
			graph_preprocess()

			(finish_time, _) = HASCH(processor_speeds)
			this_SLR = float(finish_time) / (float(max_weight_path) / processor_speeds[0])
			acc += this_SLR
		print i
		SLR.append(float(acc) / num_samples)
		num_tasks.append(i)


	if plot_results:
		plt.scatter(num_tasks, SLR, c = "Blue", marker = 'o', label = "SLR")
		plt.title("Average CHEFT SLR vs. Number of Tasks")
		plt.xlabel("Number of Tasks")
		plt.ylabel("Average SLR")
		plt.show()

def test_randomgraphs_randomweights(proc_test = [10, 10, 10, 20, 30, 50]):
	global subplot_ct
	binomial_init_graph(20, .4)
	unif_init_weights(20, 150)

	subplot_ct = 1

	plt.figure(figsize = (15, 3))

	for i in range(4):
		binomial_init_graph(20, .4)
		graph_preprocess()
		run_baseline_v_HASCH_subsets(proc_test)

	plt.legend()
	plt.xlabel("Cost of Processors")
	plt.ylabel("Makespan")
	plt.title("CHEFT vs Max Parallelism Benchmark")
	plt.show()


	subplot_ct = 1
	plt.figure(figsize = (15, 3))

	for i in range(4):
		binomial_init_graph(20, .2)
		graph_preprocess()
		run_baseline_v_HASCH_subsets(proc_test)

	plt.title("G(20, 0.2)")
	plt.show()


	subplot_ct = 1
	plt.figure(figsize = (15, 3))

	for i in range(4):
		binomial_init_graph(20, .1)
		graph_preprocess()
		run_baseline_v_HASCH_subsets(proc_test)

	plt.title("G(20, 0.1)")
	plt.show()

def test_randomweights(proc_test = [10, 15, 20]):
	global subplot_ct

	def do_run(left, right):
		global subplot_ct
		subplot_ct = 1
		plt.figure(figsize = (15, 3))

		for i in range(4):
			unif_init_weights(left, right)
			graph_preprocess()
			run_baseline_v_HASCH_subsets(proc_test)

		plt.show()

	do_run(20, 150)
	do_run(50, 70)
	do_run(10, 300)



if __name__ == "__main__":
	init_from_file()
	unif_init_weights(20, 150)
	graph_preprocess()

	#SLR_comparison([10, 15, 20, 40])

	#test_randomweights()
	
	test_randomgraphs_randomweights([10, 10, 20, 30, 40])
	#test_randomgraphs_randomweights([30, 30, 30, 30])

	
	# run_HASCH_subsets([10, 10, 10, 15, 15, 20, 40], targ_lat = 12, hyp_speed = 60)





