pub trait Selection<T> {
    fn selection(&self, size: usize) -> T;
}

pub trait Fitness {
    fn fitness(&self) -> u32;
}

pub trait Mutate {
    fn mutate(&mut self, n: usize, prob: f64);
}

pub trait Crossover {
    fn crossover(&mut self);
}
