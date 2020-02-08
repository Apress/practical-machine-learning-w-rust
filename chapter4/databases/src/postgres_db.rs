use std::error::Error;

use postgres;
use postgres::{Connection, TlsMode};

#[derive(Debug)]
struct Weather {
    id: i32,
    month: String,
    normal: f64,
    warmest: f64,
    coldest: f64
}
pub fn run() -> Result<(), Box<dyn Error>> {
    let conn = Connection::connect("postgresql://postgres:postgres@localhost:5432/postgres",
                                    TlsMode::None)?;

     conn.execute("CREATE TABLE IF NOT EXISTS weather (
                    id              SERIAL PRIMARY KEY,
                    month           VARCHAR NOT NULL,
                    normal          DOUBLE PRECISION NOT NULL,
                    warmest         DOUBLE PRECISION NOT NULL,
                    coldest         DOUBLE PRECISION NOT NULL
                  )", &[])?;
    let weathers = vec![
        ("January", 21.3, 27.3, 15.1),
        ("February", 23.6, 30.1, 17.0),
        ("March", 26.1, 32.7, 19.5),
        ("April", 28.0, 34.2, 21.8),
        ("May", 27.4, 33.2, 21.4),
        ("June", 24.6, 29.2, 20.1),
        ("July", 23.9, 28.1, 19.7),
        ("August", 23.5, 27.4, 19.5),
        ("September", 23.9, 28.2, 19.6),
        ("October", 23.7, 28.0, 19.3),
        ("November", 22.2, 27.0, 17.5),
        ("December", 21.1, 26.2, 16.0)
    ];

    for weather in &weathers {
        conn.execute("INSERT INTO weather (month, normal, warmest, coldest) VALUES ($1, $2, $3, $4)",
                 &[&weather.0, &weather.1, &weather.2, &weather.3])?;
    }

    for row in &conn.query("SELECT id, month, normal, warmest, coldest FROM weather", &[])? {
        let weather = Weather {
            id: row.get(0),
            month: row.get(1),
            normal: row.get(2),
            warmest: row.get(3),
            coldest: row.get(4)
        };
        println!("{:?}", weather);
    }

    // get the average value
    for row in &conn.query("SELECT AVG(warmest) FROM weather;", &[])? {
        let x: f64 = row.get(0);
        println!("{:?}", x);
    }

    Ok(())
}