// oops.rs
// $ ./oops
// Planet { co2: 0.04, nitrogen: 78.09 }
// Planet { co2: 95.32, nitrogen: 2.7 }
// For planet Planet { co2: 0.04, nitrogen: 78.09 }: co2 = 0.04, nitrogen=78.09, other_gases=21.870003
// For planet Planet { co2: 95.32, nitrogen: 2.7 }: co2 = 95.32, nitrogen=2.7, other_gases=1.9800003

#[derive(Debug)]
struct Planet {
    co2: f32,
    nitrogen: f32
}

trait Atmosphere {
    fn new(co2: f32, nitrogen: f32) -> Self;
    fn amount_of_other_gases(&self) -> f32;
    fn summarize(&self);
}

impl Atmosphere for Planet {
    fn new(co2: f32, nitrogen: f32) -> Planet {
        Planet { co2: co2, nitrogen: nitrogen }
    }

    fn amount_of_other_gases(&self) -> f32 {
        100.0 - self.co2 - self.nitrogen
    }

    fn summarize(&self) {
        let other_gases = self.amount_of_other_gases();
        println!("For planet {planet:?}: co2 = {co2}, nitrogen={nitrogen}, other_gases={other_gases}",
            planet=self, co2=self.co2, nitrogen=self.nitrogen, other_gases=other_gases);
    }
}

fn main() {
    let earth = Planet { co2: 0.04, nitrogen: 78.09 };
    println!("{:?}", earth);

    let mars = Planet { co2: 95.32, nitrogen: 2.7 };
    println!("{:?}", mars);

    earth.summarize();

    mars.summarize();
}
