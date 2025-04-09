use core::f64;
use rayon::prelude::*;

use crate::genetic::{Crossover, Fitness, Mutate, Selection};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, rng};

#[derive(Debug, Clone, serde::Deserialize)]
struct Thing {
    name: String,
    x: f64,
    y: f64,
}

impl Thing {
    fn new(name: &str, x: f64, y: f64) -> Self {
        Self {
            name: name.to_string(),
            x,
            y,
        }
    }

    fn distance(&self, other: &Thing) -> f64 {
        let x = self.x - other.x;
        let y = self.y - other.y;
        f64::sqrt(x * x + y * y)
    }
}

#[derive(Debug, Clone)]
struct Genome {
    data: Vec<usize>,
    things: Vec<Thing>,
}

impl Genome {
    fn new(things: &[Thing]) -> Self {
        let mut rng = rng();
        let mut data: Vec<usize> = (0..things.len()).collect();
        data.shuffle(&mut rng);
        Self {
            data,
            things: things.to_owned(),
        }
    }
}

impl Fitness<f64> for Genome {
    fn fitness(&self) -> f64 {
        let mut distance: f64 = 0.0;

        let mut a = self.things.get(*self.data.first().unwrap()).unwrap();
        let mut b = self.things.get(*self.data.last().unwrap()).unwrap();
        distance += a.distance(b);

        for w in self.data.windows(2) {
            a = self.things.get(*w.to_owned().first().unwrap()).unwrap();
            b = self.things.get(*w.to_owned().last().unwrap()).unwrap();
            distance += a.distance(b);
        }

        distance
    }
}

impl Mutate for Genome {
    fn mutate(&mut self, n: usize, prob: f64) {
        let mut rng = rng();
        let mut count = 0;
        while n < count {
            if rng.random_bool(prob) {
                let index: Vec<usize> = self.data.choose_multiple(&mut rng, 2).cloned().collect();
                self.data
                    .swap(*index.first().unwrap(), *index.last().unwrap());
                count += 1;
            }
        }
    }
}

impl Eq for Genome {}

impl Ord for Genome {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fitness().total_cmp(&other.fitness())
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

#[derive(Debug, Clone)]
struct Population {
    data: Vec<Genome>,
}

impl Population {
    fn new(pop_size: u32, things: &[Thing]) -> Self {
        let data = (0..pop_size).map(|_| Genome::new(things)).collect();

        Self { data }
    }
}

impl Selection for Population {
    fn selection(&self, size: usize) -> Self {
        let mut rng = rng();
        let worst = self.data.last().unwrap().fitness();
        let data: Vec<Genome> = self
            .data
            .choose_multiple_weighted(&mut rng, size, |genome| worst - genome.fitness() + 1.0)
            .unwrap()
            .cloned()
            .collect();
        Self { data }
    }
}

struct Pair<'a> {
    a: &'a mut Genome,
    b: &'a mut Genome,
}

impl Crossover for Pair<'_> {
    fn crossover(&mut self) {
        let mut rng = rng();
        let length = self.a.data.len();
        let cut_point = rng.random_range(0..length);
        let mut new_a = self.a.data[0..cut_point].to_vec();
        let mut new_b = self.b.data[0..cut_point].to_vec();
        self.b.data.iter().for_each(|x| {
            if !new_a.contains(x) {
                new_a.push(*x)
            }
        });
        self.a.data.iter().for_each(|x| {
            if !new_b.contains(x) {
                new_b.push(*x)
            }
        });
        self.a.data = new_a;
        self.b.data = new_b;
    }
}

impl Mutate for Pair<'_> {
    fn mutate(&mut self, n: usize, prob: f64) {
        self.a.mutate(n, prob);
        self.b.mutate(n, prob);
    }
}

fn run_evolution(
    population: &mut Population,
    target: f64,
    generation_limit: usize,
) -> Option<(&Genome, usize)> {
    for i in 0..generation_limit {
        population.data.sort();
        if population.data.first().unwrap().fitness() <= target && target > 0.0 {
            return Some((population.data.first().unwrap(), i));
        }

        let mut new_population = population.clone();
        new_population.data = new_population
            .data
            .get(0..population.data.len() / 2)
            .unwrap()
            .to_vec();
        for _ in (0..population.data.len()).step_by(4) {
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

        // dbg!(
        //     "generation {} best: {:?} from population of {}",
        //     i,
        //     new_population.data.first().unwrap().fitness(),
        //     new_population.data.len(),
        // );
        *population = new_population;
    }

    Some((population.data.first().unwrap(), 0))
}

fn read_csv(path: &str) -> Vec<Thing> {
    let mut rdr = csv::Reader::from_path(path).expect("failed to open csv");
    rdr.deserialize()
        .map(|result| {
            let record: Thing = result.unwrap();
            record
        })
        .collect::<Vec<Thing>>()
}

pub fn run() {
    let things = read_csv("data/uk-cities.csv");
    let mut distances: Vec<Genome> = vec![];
    (0..100)
        .into_par_iter()
        .map(|_| {
            let mut population = Population::new(300, &things);
            let solution = run_evolution(&mut population, 50.0, 100).expect("no solution found");
            println!("solution: {} - {:?}", solution.0.fitness(), solution.0.data);
            solution.0.clone()
        })
        .collect_into_vec(&mut distances);

    let mut min = f64::MAX;
    let mut best = distances.first().unwrap().to_owned();
    for d in distances {
        let dist = d.fitness();
        if dist < min {
            min = dist;
            best = d
        }
    }

    println!("best solution was: {} - {:?}", min, best.data)
}
