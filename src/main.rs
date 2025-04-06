use rand::{Rng, rng, seq::IndexedRandom};

#[derive(Debug, Clone)]
struct Thing {
    _name: String,
    value: u32,
    weight: u32,
}

impl Thing {
    fn new(name: &str, value: u32, weight: u32) -> Thing {
        Thing {
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
    fn new(pop_size: u32, things: &[Thing], weight_limit: u32) -> Population {
        let data = (0..pop_size)
            .map(|_| Genome::new(things, weight_limit))
            .collect();

        Population { data }
    }

    fn selection(&self) -> Population {
        let mut rng = rng();
        let data: Vec<Genome> = self
            .data
            .choose_multiple_weighted(&mut rng, 2, |genome| genome.fitness())
            .unwrap()
            .cloned()
            .collect();
        Population { data }
    }
}

#[derive(Debug, Clone)]
struct Genome {
    data: Vec<u32>,
    things: Vec<Thing>,
    weight_limit: u32,
}

impl Genome {
    fn new(things: &[Thing], weight_limit: u32) -> Genome {
        let mut rng = rng();
        let data = (0..things.len()).map(|_| rng.random_range(0..=1)).collect();
        Genome {
            data,
            things: things.to_owned(),
            weight_limit,
        }
    }

    fn fitness(&self) -> u32 {
        let mut weight = 0;
        let mut value = 0;

        self.things
            .iter()
            .enumerate()
            .map_while(|(i, thing)| {
                if weight + self.data[i] * thing.weight <= self.weight_limit {
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

fn single_point_crossover(pair: &mut [Genome]) -> (Genome, Genome) {
    let mut a = pair.first().unwrap().clone();
    let mut b = pair.last().unwrap().clone();
    let mut rng = rng();
    let length = a.data.len();
    let cut_point = rng.random_range(0..length);
    let a_swap = a.data.split_off(cut_point);
    let b_swap = b.data.split_off(cut_point);
    a.data = [a.data.clone(), b_swap].concat();
    b.data = [b.data.clone(), a_swap].concat();
    (a.clone(), b.clone())
}

fn run_evolution(
    population: &mut Population,
    target: u32,
    crossover_fn: fn(p: &mut [Genome]) -> (Genome, Genome),
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
            let mut parents = population.selection();
            let pair = parents.data.get_mut(0..=1).unwrap();
            let mut children = crossover_fn(pair);
            children.0.mutate(1, 0.5);
            children.1.mutate(1, 0.5);
            new_population.data.push(children.0);
            new_population.data.push(children.1);
        }

        *population = new_population;
    }

    None
}

fn main() {
    let weight_limit = 3000;
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

    let mut population = Population::new(10, &things, weight_limit);
    let solution = run_evolution(&mut population, 1310, single_point_crossover, 1000)
        .expect("no solution found");

    println!(
        "{} -- {:?} -- {:?}",
        solution.1,
        solution.0.fitness(),
        solution.0.data
    );
    println!("{:?}", solution.0.things);
}
