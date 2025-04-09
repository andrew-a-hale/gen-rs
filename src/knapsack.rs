use crate::genetic::{self, Crossover, Fitness, Mutate, Selection};
use rand::{Rng, rng, seq::IndexedRandom};

#[derive(Debug, Clone)]
struct Thing {
    _name: String,
    value: u32,
    weight: u32,
}

impl Thing {
    fn new(name: &str, value: u32, weight: u32) -> Self {
        Self {
            _name: name.to_string(),
            value,
            weight,
        }
    }
}

#[derive(Debug, Clone)]
struct Population {
    data: Vec<Genome>,
}

impl Population {
    fn new(pop_size: u32, things: &[Thing], limit: u32) -> Self {
        let data = (0..pop_size).map(|_| Genome::new(things, limit)).collect();

        Self { data }
    }
}

impl genetic::Selection for Population {
    fn selection(&self, size: usize) -> Self {
        let mut rng = rng();
        let data: Vec<Genome> = self
            .data
            .choose_multiple_weighted(&mut rng, size, |genome| genome.fitness())
            .unwrap()
            .cloned()
            .collect();
        Self { data }
    }
}

#[derive(Debug, Clone)]
struct Genome {
    data: Vec<u32>,
    things: Vec<Thing>,
    limit: u32,
}

impl Genome {
    fn new(things: &[Thing], limit: u32) -> Self {
        let mut rng = rng();
        let data = (0..things.len()).map(|_| rng.random_range(0..=1)).collect();
        Self {
            data,
            things: things.to_owned(),
            limit,
        }
    }
}

impl genetic::Fitness<u32> for Genome {
    fn fitness(&self) -> u32 {
        let mut weight = 0;
        let mut value = 0;

        self.things
            .iter()
            .enumerate()
            .map_while(|(i, thing)| {
                if weight + self.data[i] * thing.weight <= self.limit {
                    weight += self.data[i] * thing.weight;
                    value += self.data[i] * thing.value;
                    Some((weight, value))
                } else {
                    None
                }
            })
            .for_each(|_| {});

        value
    }
}

impl genetic::Mutate for Genome {
    fn mutate(&mut self, n: usize, prob: f64) {
        let mut count = 0;
        let mut rng = rng();
        while count < n {
            let i = rng.random_range(0..self.data.len());
            if let Some(x) = self.data.get_mut(i) {
                if rng.random_bool(prob) {
                    *x = (*x + 1) % 2;
                    count += 1;
                }
            }
        }
    }
}

struct Pair<'a> {
    a: &'a mut Genome,
    b: &'a mut Genome,
}

impl genetic::Crossover for Pair<'_> {
    fn crossover(&mut self) {
        let mut rng = rng();
        let length = self.a.data.len();
        let cut_point = rng.random_range(0..length);
        let a_swap = self.a.data.split_off(cut_point);
        let b_swap = self.b.data.split_off(cut_point);
        self.a.data = [self.a.data.clone(), b_swap].concat();
        self.b.data = [self.b.data.clone(), a_swap].concat();
    }
}

impl genetic::Mutate for Pair<'_> {
    fn mutate(&mut self, n: usize, prob: f64) {
        self.a.mutate(n, prob);
        self.b.mutate(n, prob);
    }
}

impl Eq for Genome {}

impl Ord for Genome {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.fitness().cmp(&self.fitness())
    }
}

impl PartialEq for Genome {
    fn eq(&self, other: &Self) -> bool {
        self.fitness() == other.fitness()
    }
}

impl PartialOrd for Genome {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn run_evolution(
    population: &mut Population,
    target: u32,
    generation_limit: usize,
) -> Option<(&Genome, usize)> {
    for i in 0..generation_limit {
        population.data.sort();
        if population.data.first().unwrap().fitness() >= target {
            return Some((population.data.first().unwrap(), i));
        }

        let mut new_population = population.clone();
        new_population.data = new_population.data.get(0..=1).unwrap().to_vec();

        for _ in (0..population.data.len()).step_by(2) {
            let parents = population.selection(2);
            let mut a = parents.data.first().as_mut().unwrap().clone();
            let mut b = parents.data.last().as_mut().unwrap().clone();
            let mut pair = Pair {
                a: &mut a,
                b: &mut b,
            };
            pair.crossover();
            pair.mutate(1, 0.5);
            new_population.data.push(pair.a.to_owned());
            new_population.data.push(pair.b.to_owned());
        }

        *population = new_population;
    }

    None
}

pub fn run() {
    let limit = 3000;
    let things = vec![
        Thing::new("Laptop", 500, 2200),
        Thing::new("Headphones", 150, 160),
        Thing::new("Coffee Mug", 60, 350),
        Thing::new("Notepad", 40, 333),
        Thing::new("Water Bottle", 30, 192),
        Thing::new("Mints", 5, 25),
        Thing::new("Socks", 10, 38),
        Thing::new("Tissues", 15, 80),
        Thing::new("Phone", 500, 200),
        Thing::new("Baseball Cap", 100, 70),
    ];

    let mut population = Population::new(10, &things, limit);
    let solution = run_evolution(&mut population, 1310, 1000).expect("no solution found");

    println!(
        "{} -- {:?} -- {:?}",
        solution.1,
        solution.0.fitness(),
        solution.0.data
    );
    println!("{:?}", solution.0.things);
}
