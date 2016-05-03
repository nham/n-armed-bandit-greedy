#![feature(append)]

extern crate rand;

use rand::Rng;
use rand::distributions::normal::{Normal, StandardNormal};
use rand::distributions::{IndependentSample, Range};
use std::fs::File;
use std::io::{self, Write};

struct EpsilonGreedyBandit {
    // number of arms
    n: usize,

    // Actions are a_0 through a_{n-1}. Each has
    // a vector of past rewards received when choosing that action.
    // All past rewards for a given action a_k are averaged
    // to obtain an estimate of Q_t(a), the value of taking
    // action a at time t, which is not known with certainty.
    past_rewards: Vec<Vec<f64>>,

    // parameter for the greediness of the bandit
    epsilon: f64,
}

impl EpsilonGreedyBandit {
    fn new(n: usize, epsilon: f64) -> EpsilonGreedyBandit {
        let mut past_rewards = Vec::new();
        for _ in 0..n {
            past_rewards.push(vec![]);
        }

        EpsilonGreedyBandit {
            n: n,
            past_rewards: past_rewards,
            epsilon: epsilon,
        }
    }

    fn choose_action(&self) -> usize {
        // It doesn't make sense if there are no possible actions.
        // If there's only one possible action, the whole exercise is
        // pointless, but we still allow it.
        assert!(self.n > 0);

        // estimate "true values" for each action
        let mut estimates = Vec::new();
        for i in 0..self.n {
            estimates.push(self.calculate_estimate(i));
        }

        // Pick a random number uniformly between 0 and 1 to see
        // if it's > epsilon (and so pick a greedy action)
        // or <= (and so pick a non-greedy move)
        let between = Range::new(0f64, 1.);
        let x = between.ind_sample(&mut rand::thread_rng());

        if x > self.epsilon {
            // choose an action with a max value
            let mut max_actions = vec![0];
            let mut max_value = estimates[0];
            for i in 1..self.n {
                if estimates[i] > max_value {
                    max_actions.clear();
                    max_actions.push(i);
                    max_value = estimates[i];
                } else if estimates[i] == max_value {
                    max_actions.push(i);
                }
            }
            assert!(max_actions.len() > 0);
            let k = rand::thread_rng().gen_range(0, max_actions.len());
            max_actions[k]
        } else {
            // choose a non-max action
            let mut non_max_actions = vec![];
            let mut max_actions = vec![0];
            let mut max_value = estimates[0];
            for i in 1..self.n {
                if estimates[i] > max_value {
                    non_max_actions.append(&mut max_actions);
                    max_actions.push(i);
                    max_value = estimates[i];
                } else if estimates[i] == max_value {
                    max_actions.push(i);
                } else {
                    non_max_actions.push(i);
                }
            }
            if non_max_actions.len() > 0 {
                let k = rand::thread_rng().gen_range(0, non_max_actions.len());
                non_max_actions[k]
            } else {
                let k = rand::thread_rng().gen_range(0, max_actions.len());
                max_actions[k]
            }

        }
    }

    fn receive_reward(&mut self, reward: f64, action: usize) {
        self.past_rewards[action].push(reward);
    }

    fn calculate_estimate(&self, action: usize) -> f64 {
        let num_actions = self.past_rewards.len();
        assert!(action < num_actions);

        let num_past_rewards = self.past_rewards[action].len();
        if num_past_rewards == 0 { return 0.0 }

        let mut sum = 0.0;
        for i in 0..num_past_rewards {
            sum += self.past_rewards[action][i];
        }

        sum / (num_past_rewards as f64)
    }
}

struct BanditTask {
    n: usize,
}

impl BanditTask {
    fn new(n: usize) -> BanditTask {
        BanditTask {
            n: n,
        }
    }

    // Returns vector of the reward at each stage
    fn run_task(&mut self, bandit: &mut EpsilonGreedyBandit, num_plays: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut rewards = vec![];

        let mut q_star: Vec<f64> = vec![];
        for i in 0..self.n {
            let StandardNormal(true_value) = rand::random();
            q_star.push(true_value);
        }

        for i in 0..num_plays {
            // For each task i and each action j, we pick Q_i^*(j), the "true value"
            // of action j during task i. This is picked from a standard normal dist.
            //
            // Then the reward for selecting action j during task i is chosen from
            // a normal with mean Q_i^*(a) and variance 1
            let mut reward: Vec<f64> = vec![];
            for j in 0..self.n {
                // Normal with mean q_star and variance 1
                let normal = Normal::new(q_star[j], 1.0); 
                reward.push( normal.ind_sample(&mut rng) );
            }

            // Bandit is prompted to choose an action, 
            let action = bandit.choose_action();
            rewards.push(reward[action]);
            bandit.receive_reward(reward[action], action);
        }
        rewards
    }
}

fn dump_vec_to_file(v: &Vec<f64>, file_name: &str) -> io::Result<()> {
    let mut f = try!(File::create(file_name));
    for i in 0..v.len() {
        let s = format!("{:?}", v[i]);
        try!(f.write(s.as_bytes()));
        try!(f.write(b"\n"));
    }
    Ok(())
}

fn main() {
    println!("Hello, world!");
    let n = 10;
    let num_tasks = 2000;
    let num_plays = 1000;
    let epsilon = 0.2;

    let mut avg_rewards = vec![];
    for _ in 0..num_plays {
        avg_rewards.push(0.0);
    }

    for i in 0..num_tasks {
        println!("Task #{}", i);
        let mut task = BanditTask::new(n);
        let mut bandit = EpsilonGreedyBandit::new(n, epsilon);
        let rewards = task.run_task(&mut bandit, num_plays);

        for i in 0..num_plays {
            avg_rewards[i] += rewards[i];
        }
    }

    for i in 0..num_plays {
        avg_rewards[i] /= num_plays as f64;
    }

    dump_vec_to_file(&avg_rewards, "eps_0_2.dat");
}
