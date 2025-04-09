pub trait Selection {
    fn selection(&self, size: usize) -> Self;
}

pub trait Fitness<T> {
    fn fitness(&self) -> T;
}

pub trait Mutate {
    fn mutate(&mut self, n: usize, prob: f64);
}

pub trait Crossover {
    fn crossover(&mut self);
}
