// enumerations.rs

#[derive(Debug)]
enum NationalHolidays {
    GandhiJayanti,
    RepublicDay,
    IndependenceDay,
}

fn inspect(day: NationalHolidays) -> String {
    match day {
        NationalHolidays::GandhiJayanti => String::from("Oct 2"),
        NationalHolidays::RepublicDay => String::from("Jan 26"),
        NationalHolidays::IndependenceDay => String::from("Aug 15"),
    }
}

fn main() {
    let day = NationalHolidays::GandhiJayanti;
    let date = inspect(day);
    println!("{:?}", date); // output: Oct 2
}