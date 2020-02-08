use std::time::{SystemTime, UNIX_EPOCH};
use reqwest;
use scraper::{Selector, Html};


fn main() -> Result<(), Box<std::error::Error>> {
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    let mut resp = reqwest::get(
        "https://www.moneycontrol.com/india/stockpricequote/power-generation-distribution/ntpc/NTP")?;
    assert!(resp.status().is_success());

    let body = resp.text().unwrap();
    let fragment = Html::parse_document(&body);
    // let stories = Selector::parse("#Bse_Prc_tick > strong:nth-child(1)").unwrap();
    let stories = Selector::parse("#mktdet_2 > div:nth-child(2) > div:nth-child(3) > div.FR.gD_12").unwrap();

    for price in fragment.select(&stories) {
        let price_txt = price.text().collect::<Vec<_>>();
        if price_txt.len() == 1 {
            println!("{:?}", (since_the_epoch, price_txt[0]));
        }
    }

    Ok(())
}

