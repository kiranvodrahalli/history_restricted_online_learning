# Experiments for NeurIPS Submission "History-Restricted Online Learning"
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style('white')
from copy import deepcopy
from random import randrange, choices


# Save Location for Files
loc = '~/history_restricted_experiments/'

#### Algorithms 

# regular MW
def MW(rewards, eta, M=None):
	T = len(rewards)
	d = len(rewards[0])
	w = np.ones(d)
	w_seq = []
	w_seq.append(deepcopy(w))
	for t in range(T):
		r_t = rewards[t]
		for i in range(d):
			w[i] = w[i]*(1 + eta*r_t[i])
		w_seq.append(deepcopy(w))
	return w, w_seq


# smooth (high regret) M-history-restricted MW
# e.g., mean-based history restricted 
def smooth_HR_MW(rewards, eta, M):
	T = len(rewards)
	d = len(rewards[0])
	w = np.ones(d)
	w_seq = []
	w_seq.append(deepcopy(w))
	for t in range(T):
		r_t = rewards[t]
		for i in range(d):
			if t < M:
				w[i] = w[i]*(1 + eta*r_t[i])
			else:
				w[i] = w[i]*((1 + eta*r_t[i])/(1 + eta*rewards[t - M][i]))
		w_seq.append(deepcopy(w))
	return w, w_seq


# restart (low-regret) M-history-restricted MW
# periodic restart algorithm
def restart_HR_MW(rewards, eta, M):
	T = len(rewards)
	d = len(rewards[0])
	w = np.ones(d)
	w_seq = []
	w_seq.append(deepcopy(w))
	for t in range(T):
		r_t = rewards[t]
		# reset mod M (requires knowing t)
		if t % M == 0:
			w = np.ones(d)
		# MW update
		for i in range(d):
			w[i] = w[i]*(1 + eta*r_t[i])
		w_seq.append(deepcopy(w))

	return w, w_seq


def avg_restart_HR_MW(rewards, eta, M):
	T = len(rewards)
	d = len(rewards[0])
	w = np.ones(d)
	w_seq = []
	w_seq.append(deepcopy(w))
	for t in range(T):
		# random index to start running MW from in the past
		j = randrange(M) # 0 through M - 1
		# j = 0 corresponds to resetting at the current time step (like at t%M == 0 above)
		# j = M - 1 corresponds to using whole previous memory.
		w = np.ones(d)
		# MW update: update the past M rewards
		for k in range(max(0, t - j), t+1):
			r_k = rewards[k]
			for i in range(d):
				w[i] = w[i]*(1 + eta*r_k[i])
		w_seq.append(deepcopy(w))

	return w, w_seq

# avg restart (low-regret) T-history-restricted MW (not mean based)
# this should be the same across history! currently isn't,wtf??
def avg_restart_MW(rewards, eta, M=None):
	T = len(rewards)
	return avg_restart_HR_MW(rewards, eta, T)


def FTL(rewards, eta, M=None):
	T = len(rewards)
	d = len(rewards[0])
	w = np.ones(d)
	w_seq = []
	w_seq.append(deepcopy(w))
	for t in range(T):
		r_t = rewards[t]
		for i in range(d):
			w[i] = w[i]*(1 + eta*r_t[i])
		# replace with vector of 0s with only 1 on the max
		prediction = np.zeros(d) 
		prediction[np.argmax(w)] = 1
		w_seq.append(deepcopy(prediction))
	return w, w_seq



##### Evaluation Methods


# takes in reward_means (e.g., need to know the generating distribution)
def true_expected_avg_regret(w_seq, reward_means):
	T = len(reward_means)
	# calculate best in hindsight
	max_single_reward = (1./T) * np.max(sum(reward_means[t] for t in range(T)))
	curr_total_policy_reward = 0
	avg_regret_evolution = []
	total_regret_evolution = []
	for t in range(T):
		wt = w_seq[t]
		rt = reward_means[t]
		curr_total_policy_reward += np.dot(wt/np.linalg.norm(wt, 1), rt) # NORMALIZE!!!
		curr_avg_policy_reward = (1./(t + 1))*curr_total_policy_reward
		##### also compute evolution of regret
		curr_max_single_reward = (1./(t + 1))*np.max(sum(reward_means[j] for j in range(t)))
		curr_avg_regret = curr_max_single_reward - curr_avg_policy_reward
		curr_total_regret = (t + 1)*curr_max_single_reward - curr_total_policy_reward
		avg_regret_evolution.append(curr_avg_regret)
		total_regret_evolution.append(curr_total_regret)
	#print(f"avg regret evolution = {avg_regret_evolution}")
	return max_single_reward - (1./T)*curr_total_policy_reward, avg_regret_evolution, total_regret_evolution






##### Examples to Test on 

# 2 actions: 

# 1. (alpha, beta)-block style examples (correspond to the hard construction for MW) (don't really need the other types of blocks)
# 2. standard stochastic example (1/2, 1/2+eps), eps = 1/sqrt(T) (?)
# 3. "drifting" examples:
# 		- intuition is that for small history, if the reward is resetting periodically on a scale about the same as M,
#.      - it should be beneficial to "forget" the history at that same frequency. it should be less beneficial if 
#.      - the rewards are stationary for a longer time scale than the resetting frequency. 
#.      - if the rewards change at a higher frequency than history frequency, it may still be beneficial to have history-restriction
#.        compared to full large history alg. 
#.      - we can implement stationarity in a few ways:
#.        a) sample from different reward distributions whose probability of outputting certain rewards varies with time.
#.        b) Markov chain reward distribution for arms, vary the mixing time of different blocks and make it non-stationary


def flip(p, h, t):
    return h if random.random() < p else t

def drifting_coin_flip_prob(t, period):
	return np.abs(np.sin(math.pi/6 + (math.pi/period)*t))

def qp(period, T):
	time = [t for t in range(T)]
	plt.figure()
	plt.plot(time, list(map(lambda t: drifting_coin_flip_prob(t, period), time)))
	plt.xlabel('round')
	plt.ylabel(r'coin flip probability $p$')
	plt.title(f'Drifting Reward with Period {period}')
	plt.savefig(f"{loc}figures/drifting_reward_{period}.eps", format='eps', dpi=200)
	plt.close()



def plot_drifting_rewards(T):
	time = [t for t in range(T)]
	plt.figure()
	for freq in [.1, 0.2, .5, 0.8, 1, 2]:
		period = freq*T
		plt.plot(time, list(map(lambda t: drifting_coin_flip_prob(t, int(period)), time)), label=f'Drifting Reward with Period {freq}T')
	plt.xlabel('round')
	plt.ylabel(r'coin flip probability $p$')
	plt.title(f'Example Drifting Rewards over Timescale {T}')
	plt.legend()
	plt.show()
	#plt.savefig(f"{loc}figures/drifting_rewards.eps", format='eps', dpi=200)
	#plt.close()


def iid_stochastic_reward(T):
	rewards = []
	for t in range(T):
		arm1 = flip(0.5 + 1/np.sqrt(T), 1, -1)
		arm2 = flip(0.5, 1, -1)
		rewards.append(np.array([arm1, arm2]))
	return np.array(rewards)

def iid_stochastic_reward_mean(T):
	reward_means = []
	for t in range(T):
		arm1 = 1*(0.5 + 1/np.sqrt(T)) - (0.5 - 1/np.sqrt(T))
		arm2 = 0.5 - 0.5
		reward_means.append(np.array([arm1, arm2]))
	return np.array(reward_means)


def random_walk_reward_and_mean(T, stddev=0.1):
	rewards = []
	reward_means = []
	p = 0.5
	for t in range(T):
		arm1 = flip(p, 1, -1)
		mean1 = 1*p - (1 - p)
		arm2 = flip(0.5, 1, -1)
		mean2 = 0
		rewards.append(np.array([arm1, arm2]))
		reward_means.append(np.array([mean1, mean2]))
		# update coin flip
		p += np.random.normal(0, stddev)
		# needs to be normalized to be in 0, 1
		p = max(min(p, 1), 0)
	return np.array(rewards), np.array(reward_means)

def random_walk_ps(T, stddev=0.1):
	rewards = []
	reward_means = []
	p = 0.5
	ps = [p]
	for t in range(T):
		arm1 = flip(p, 1, -1)
		mean1 = 1*p - (1 - p)
		arm2 = flip(0.5, 1, -1)
		mean2 = 0
		rewards.append(np.array([arm1, arm2]))
		reward_means.append(np.array([mean1, mean2]))
		# update coin flip 
		p += np.random.normal(0, stddev)
		# needs to be normalized to be in 0, 1
		p = max(min(p, 1), 0)
		ps.append(p)
	return np.array(rewards), np.array(reward_means), ps

def plot_randomwalk_rewards(T):
	time = [t for t in range(T)]
	plt.figure()
	for sigma in [0.0001, 0.001, 0.01, 0.02]:
		_, _, ps = random_walk_ps(T, stddev=sigma)
		plt.plot(time, ps[:T], label=f'Random Walk Reward with Standard Deviation {sigma}')
	plt.xlabel('round')
	plt.ylabel(r'coin flip probability $p$')
	plt.title(f'Example Random Walk Rewards over Timescale {T}')
	plt.legend()
	#plt.savefig(f"{loc}figures/randomwalk_rewards.eps", format='eps', dpi=200)
	#plt.close()
	plt.show()


# can treat each arm as sampling a coin with bias p and changing p periodically. 
# "best arm" changes periodically basically. periodic mean, but still stochastic!
# have to really be careful with what we choose the period to be. 
# have to choose period so that it's on par with T to get something useful.
# otherwise, the probability changes a tiny amount. 

# we split up the expected reward calculation given a sequence of probability vector predictions w_t:
# we still generate the w_t from the random reward, but computing expectation (given that we know it's a coin with a changing bias)
# is easy to do directly without sampling.
def drifting_reward(T, period1, period2):
	rewards = []
	# absolute value halves usual period
	for t in range(T):
		arm1 = flip(drifting_coin_flip_prob(t, period1), 1, -1)
		arm2 = flip(drifting_coin_flip_prob(t, period2), 1, -1)
		rewards.append(np.array([arm1, arm2]))
	return np.array(rewards)

def drifting_reward_mean(T, period1, period2):
	reward_means = []
	# absolute value halves usual period
	for t in range(T):
		arm1 = 1*drifting_coin_flip_prob(t, period1) - (1 - drifting_coin_flip_prob(t, period1))
		arm2 = 1*drifting_coin_flip_prob(t, period2)- (1 - drifting_coin_flip_prob(t, period2))
		reward_means.append(np.array([arm1, arm2]))
	return np.array(reward_means)


def simple_drifting_reward(T, period):
	rewards = []
	# absolute value halves usual period
	for t in range(T):
		arm1 = flip(drifting_coin_flip_prob(t, period), 1, -1)
		arm2 = flip(0.5, 1, -1)
		rewards.append(np.array([arm1, arm2]))
	return np.array(rewards)

def simple_drifting_reward_mean(T, period):
	reward_means = []
	# absolute value halves usual period
	for t in range(T):
		arm1 = 1*drifting_coin_flip_prob(t, period) - (1 - drifting_coin_flip_prob(t, period))
		arm2 = 0.5 - 0.5
		reward_means.append(np.array([arm1, arm2]))
	return np.array(reward_means)



# fixed M = (1/3)T for simplicity
# from mean-based lower bound proof
def adversarial_history_reward(M):
	T = 3*M
	reward_means = []
	for t in range(T):
		if t <= M:
			arm1 = 1
			arm2 = 0
		elif M < t and t <= 5*M/3:
			arm1 = 0
			arm2 = 1
		elif 5*M/3 < t and t <= 2*M:
			arm1 = 1
			arm2 = 0
		elif 2*M < t:
			arm1 = 0
			arm2 = 0
		reward_means.append(np.array([arm1, arm2]))
	return np.array(reward_means)




##### Experiments

T = 1000


def compare_algs(M, eta=0.5, rewards_list=REWARDS_LIST, alg_list=ALG_LIST):
	results_dict = {}
	for alg_name, alg in alg_list:
		if alg_name not in results_dict:
			results_dict[alg_name] = {}
		for rewards_name, rewards, reward_means in rewards_list:
			_, w_seq = alg(rewards, eta, M)
			avg_regret, avg_regret_evolution, total_regret_evolution = true_expected_avg_regret(w_seq, reward_means)
			results_dict[alg_name][rewards_name] = (avg_regret, avg_regret_evolution, total_regret_evolution, eta, w_seq)

	return results_dict

# average number of runs = 3
def compare_algs_avg_runs(M, num_runs=3, eta=0.5, T=T, rewards_list=REWARDS_LIST, alg_list=ALG_LIST):
	results_dict = {}
	for alg_name, alg in alg_list:
		if alg_name not in results_dict:
			results_dict[alg_name] = {}
		for rewards_name, rewards, reward_means in rewards_list:
			avg_regret = 0
			avg_regret_evolution = np.zeros(T)
			total_regret_evolution = np.zeros(T)
			# averaging over the randomness in the alg
			for k in range(num_runs):
				_, w_seq = alg(rewards, eta, M)
				avg_regret_k, avg_regret_evolution_k, total_regret_evolution_k = true_expected_avg_regret(w_seq, reward_means)
				avg_regret += (1./num_runs)*avg_regret_k 
				avg_regret_evolution += (1./num_runs)*np.array(avg_regret_evolution_k)
				total_regret_evolution += (1./num_runs)*np.array(total_regret_evolution_k)
			results_dict[alg_name][rewards_name] = (avg_regret, avg_regret_evolution, total_regret_evolution, eta)

	return results_dict











ALG_LIST = [("MW", MW), ("History-Restricted Mean-Based MW", smooth_HR_MW), ("Periodic Restart History-Restricted MW", restart_HR_MW), ("Average Restart History-Restricted MW", avg_restart_HR_MW), ("Full-Horizon Average Restart MW", avg_restart_MW)]
NEW_ALG_LIST = ALG_LIST
all_frequencies = [0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 1, 2]
random_walk_stddevs = [0.0001, 0.001, 0.01, 0.02] # [0.01, 0.05, 0.1, 0.2] -- too large
history_values = np.linspace(0.01, 0.99, 20) 

RANDOMWALK_REWARDS_LIST = []

FULL_REWARDS_LIST = [("Stochastic Reward", iid_stochastic_reward(T), iid_stochastic_reward_mean(T))]

for stddev in random_walk_stddevs:
	# sample 3 random walks for each standard deviation
	for k in range(3):
		rewards, means = random_walk_reward_and_mean(T, stddev=stddev)
		simple_random_walk_k = (f"Random Walk Reward with Standard Deviation {stddev} #{k+1}", rewards, means)
		FULL_REWARDS_LIST.append(simple_random_walk_k)
		RANDOMWALK_REWARDS_LIST.append(simple_random_walk_k)


for i in range(len(all_frequencies)):
	fi = all_frequencies[i]
	simple_periodic = (f"Drifting Reward of Period {fi}T", simple_drifting_reward(T, T*fi), simple_drifting_reward_mean(T, T*fi))
	FULL_REWARDS_LIST.append(simple_periodic)
	for j in range(i+1, len(all_frequencies)):
		fj = all_frequencies[j]
		if fj != 1 and fi != 1:
			pairwise_periodic = (f"Drifting Rewards with Periods {fi}T and {fj}T", drifting_reward(T, fi*T, fj*T), drifting_reward_mean(T, fi*T, fj*T))
		elif fj == 1:
			pairwise_periodic = (f"Drifting Rewards with Periods {fi}T and T", drifting_reward(T, fi*T, fj*T), drifting_reward_mean(T, fi*T, fj*T))
		elif fi == 1:
			pairwise_periodic = (f"Drifting Rewards with Periods T and {fj}T", drifting_reward(T, fi*T, fj*T), drifting_reward_mean(T, fi*T, fj*T))
		FULL_REWARDS_LIST.append(pairwise_periodic)

ADVERSARIAL_REWARDS_LIST = [(f'adversarial reward for M = T/3', adversarial_history_reward(333), adversarial_history_reward(333))]
MEAN_ADVERSARIAL_REWARDS_DICT = {f'adversarial reward for M = T/3': adversarial_history_reward(333)}

def run_adversarial_experiment(bestorworsteta='0.5'):
	results_dict = {}
	M = 1./3.
	m = int(M*T)
	print(f"m={m}")
	if bestorworsteta == 'best':
		results_dict[m] = compare_algs_bestcase_eta(m, rewards_list=ADVERSARIAL_REWARDS_LIST, alg_list=ALG_LIST) 
	elif bestorworsteta == 'worst':
		results_dict[m] = compare_algs_worstcase_eta(m, rewards_list=ADVERSARIAL_REWARDS_LIST, alg_list=ALG_LIST)
	else:
		# assume bestorworsteta is a float value of eta
		results_dict[m] = compare_algs(m, float(bestorworsteta), rewards_list=ADVERSARIAL_REWARDS_LIST, alg_list=ALG_LIST)
	np.savez(f"{loc}exp_{bestorworsteta}_eta_adversarial_rewards_dict.npz", experiment=results_dict)
	return results_dict

# num_runs =  number of times we run the algorithm (averaging over randomness in alg)
def run_nonadversarial_experiments(num_runs=3, bestorworsteta='0.5'):
	results_dict = {}
	for M in history_values:
		m = int(M*T)
		print(f"m={m}")
		# assume bestorworsteta is a float value of eta
		results_dict[m] = compare_algs_avg_runs(m, num_runs=num_runs, eta=float(bestorworsteta), rewards_list=FULL_REWARDS_LIST, alg_list=NEW_ALG_LIST)
	np.savez(f"{loc}exp_{bestorworsteta}_eta_full_rewards_dict.npz", experiment=results_dict)
	return results_dict



# run fixed random walk experiments
def run_random_walk_experiments(num_runs=3, bestorworsteta='0.5'):
	results_dict = {}
	for M in history_values:
		m = int(M*T)
		print(f"m={m}")
		results_dict[m] = compare_algs_avg_runs(m, num_runs=num_runs, eta=float(bestorworsteta), rewards_list=RANDOMWALK_REWARDS_LIST, alg_list=NEW_ALG_LIST)
	np.savez(f"{loc}exp_{bestorworsteta}_eta_randomwalk_rewards_dict.npz", experiment=results_dict)
	return results_dict





## Average Regret vs. History

def avg_regret_full_rewards_comparison(reward, bestorworsteta='0.5'):
	# stochastic
	#d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_dict.npz", allow_pickle=True)
	# all non-adversarial rewards
	d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_full_rewards_dict.npz", allow_pickle=True)
	# random walk rewards
	#d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_randomwalk_rewards_dict.npz", allow_pickle=True)
	d1 = d0['experiment']
	d = d1[()]
	plot_dict = {}
	alg_names = []
	for alg_name, alg in NEW_ALG_LIST:
		xs = []
		ys = []
		for m in history_values:
			M = int(m*T)
			regret = d[M][alg_name][reward][0]
			xs.append(m)
			ys.append(regret)
		plot_dict[alg_name] = (xs, ys)
		alg_names.append(alg_name)
	plt.figure()
	for alg_name in alg_names:
		xs, ys = plot_dict[alg_name]
		plt.plot(xs, ys, label=alg_name)
	plt.xlabel("History")
	plt.ylabel("Average Expected Regret")
	plt.legend()
	plt.title(rf"Average Expected Regret for {reward} across History")
	#plt.savefig(f"{loc}figures/avg_regret_{reward}_eta={bestorworsteta}.eps", format='eps', dpi=200)
	#plt.close()
	plt.show()

def gen_avg_full_rewards_regret(eta=0.5):
	for reward_name, _, _ in FULL_REWARDS_LIST:
		avg_regret_full_rewards_comparison(reward_name, str(eta))


# Heatmaps

def heatmap_full_rewards_plot(alg, reward, bestorworsteta='0.5'):
	d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_full_rewards_dict.npz", allow_pickle=True)
	d1 = d0['experiment']
	d = d1[()]

	xs = history_values #history values M as fraction of T
	ys = np.linspace(0, .999, 200) # round values (fractions of T)
	R1 = len(xs)
	R2 = len(ys)
	Z = np.zeros((R1, R2))
	X, Y = np.meshgrid(xs, ys)
	# build Z
	xdict = dict()
	ydict = dict()
	for i in range(len(xs)):
		xval = int(xs[i]*T)
		for j in range(len(ys)):
			yval = int(ys[j]*T)
			# avg reward evolution
			Z[i, j] = d[xval][alg][reward][1][yval]
	plt.figure(figsize=(8, 8))
	plt.contourf(X, Y, Z.T, 50, cmap='Reds')
	clb = plt.colorbar()
	clb.ax.set_ylabel("Average Regret", rotation=270, labelpad=25)
	plt.xlabel("History")
	plt.ylabel("Rounds")
	plt.title(f"Average Expected Regret Growth over Rounds for Varying History for {reward}")
	#plt.savefig(f"{loc}figures/heatmap_{alg}_{reward}_eta={bestorworsteta}.eps", format='eps', dpi=200)
	plt.show()


def gen_heatmaps(eta=0.5):
	for alg_name, alg in NEW_ALG_LIST:
		for reward_name, _, _ in FULL_REWARDS_LIST:
			heatmap_plot(alg_name, reward_name, str(eta))


CHOSEN_REWARDS = ["Random Walk Reward with Standard Deviation 0.02 #2", "Drifting Reward of Period 1T", "Drifting Rewards with Periods 0.4T and 0.6T", "Drifting Rewards with Periods 0.6T and 0.8T"]

def gen_chosen_heatmaps():
	for alg_name, alg in ALG_LIST:
		if alg_name != 'MW' and alg_name != 'Full-Horizon Average Restart MW':
			for reward_name in CHOSEN_REWARDS:
				heatmap_full_rewards_plot(alg_name, reward_name)




##### Plots 

# heat maps with time and history on x and y axes, color gradient = regret, 

# adversarial example plot
def Delta_t(M):
	T = 3*M
	deltats = []
	for t in range(T):
		if t <= M:
			deltat = t
		elif M < t and t <= 5*M/3:
			deltat = M - 2*(t - M)
		elif 5*M/3 < t and t <= 2*M:
			deltat = -1*M/3
		elif 2*M < t and t <= 8*M/3:
			deltat = -1*M/3 + (t - 2*M)
		elif 8*M/3 < t:
			deltat = M/3 - (t - 8*M/3)
		deltats.append(deltat)
	return deltats

def plot_Delta_t_hard_example(M):
	deltats = Delta_t(M)
	plt.figure(figsize=(8, 8))
	plt.plot(list(range(len(deltats))), deltats)
	plt.xlabel(r't')
	plt.ylabel(r'$\Delta_t$')
	plt.title(r'$\Delta_t$ over Time')
	plt.show()
	plt.savefig(f"{loc}figures/delta_t_plot.eps", format='eps', dpi=200)



def heatmap_plot(alg, reward, bestorworsteta='0.5'):
	d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_dict.npz", allow_pickle=True)
	d1 = d0['experiment']
	d = d1[()]

	xs = history_values #history values M as fraction of T
	ys = np.linspace(0, .999, 200) # round values (fractions of T)
	R1 = len(xs)
	R2 = len(ys)
	Z = np.zeros((R1, R2))
	X, Y = np.meshgrid(xs, ys)
	# build Z
	xdict = dict()
	ydict = dict()
	for i in range(len(xs)):
		xval = int(xs[i]*T)
		for j in range(len(ys)):
			yval = int(ys[j]*T)
			# avg reward evolution
			Z[i, j] = d[xval][alg][reward][1][yval]
	plt.figure(figsize=(8, 8))
	plt.contourf(X, Y, Z.T, 50, cmap='Reds')
	clb = plt.colorbar()
	clb.ax.set_ylabel("average regret", rotation=270, labelpad=25)
	plt.xlabel("History")
	plt.ylabel("Rounds")
	plt.title(f"Average Expected Regret Growth over Rounds for Varying History")
	plt.savefig(f"{loc}figures/heatmap_{alg}_{reward}_eta={bestorworsteta}.eps", format='eps', dpi=200)


def gen_heatmaps(eta):
	for alg_name, alg in ALG_LIST:
		for reward_name, _, _ in REWARDS_LIST:
			heatmap_plot(alg_name, reward_name, str(eta))



 
def single_total_regret_over_time_comparison(reward, bestorworsteta='0.5'):
	# all non-adversarial rewards
	d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_full_rewards_dict.npz", allow_pickle=True)
	# stochastic
	#d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_dict.npz", allow_pickle=True)
	# adversarial
	#d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_adversarial_rewards_dict.npz", allow_pickle=True)
	# mean periodic rewards
	#d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_periodic_rewards_dict.npz", allow_pickle=True)
	# just random walk rewards
	#d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_randomwalk_rewards_dict.npz", allow_pickle=True)
	d1 = d0['experiment']
	d = d1[()]
	plot_dict = {}
	alg_names = []
	m = history_values[3] # 0.1647
	M = int(m*T) # 164
	for alg_name, alg in ALG_LIST:
		total_regret_time_series = d[M][alg_name][reward][2]#d[M][old_alg_name][reward][2]
		plot_dict[alg_name] = total_regret_time_series
		alg_names.append(alg_name)
	fig = plt.figure()

	for alg_name in alg_names:
		regret = plot_dict[alg_name]
		plt.plot(list(range(len(regret))), regret, label=f'{alg_name}')

	plt.xlabel("Rounds")
	plt.ylabel("Total Expected Regret")

	plt.legend()
	plt.title(rf"Total Expected Regret for {reward} across Rounds for History $M = 0.15 T$")
	plt.show()
	#plt.savefig(f"{loc}figures/total_regret_{reward}_eta={bestorworsteta}.eps", format='eps', dpi=200)
	#plt.close()


def gen_single_total_regret(eta):
	for reward_name, _, _ in FULL_REWARDS_LIST:
		single_total_regret_over_time_comparison(reward_name, str(eta))

# Adversarial Example Comparison
def single_total_regret_over_time_adversarial_comparison(bestorworsteta='0.5'):
	d0 = np.load(f"{loc}exp_{bestorworsteta}_eta_adversarial_rewards_dict.npz", allow_pickle=True)
	d1 = d0['experiment']
	d = d1[()]
	plot_dict = {}
	alg_names = []
	m = 1/3 
	M = int(m*T) # 333
	for alg_name, alg in ALG_LIST:
		if alg_name != "FTL":
			old_alg_name = alg_name
			if alg_name == "history-restricted mean-based MW":
				old_alg_name = 'smooth history-restricted MW'
			elif alg_name == "periodic restart history-restricted MW":
				old_alg_name = 'restart history-restricted MW'
			elif alg_name == "full-horizon average restart MW":
				old_alg_name = 'average restart MW'
			total_regret_time_series = d[M][old_alg_name]['adversarial reward for M = T/3'][2]
			plot_dict[alg_name] = total_regret_time_series
			alg_names.append(alg_name)
	fig = plt.figure()

	for alg_name, alg in ALG_LIST:
		if alg_name != 'FTL':
			regret = plot_dict[alg_name]
			plt.plot(list(range(len(regret))), regret, label=f'{alg_name}')

	plt.xlabel("Rounds")
	plt.ylabel("Total Expected Regret")

	plt.legend()
	plt.title(rf"Total Expected Regret for Adversarial Reward across Rounds for History $M = T/3$")
	plt.show()
	#plt.savefig(f"{loc}figures/total_regret_adversarial_eta={bestorworsteta}.eps", format='eps', dpi=200)
	#plt.close()



