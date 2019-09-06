**Verify**
----

* **URL**

  _/api/verify_

* **Method:**

  `POST`

 
* **Data Params**

   **Required:**
 
   `baseline_url=[URL]`
   `updated_url=[URL]`
   `max_depth=[integer]`
   `max_urls=[integer]`
   `prefix=[string]`

   **Optional:**
 
   `auth_baseline_username=[string]`
   `auth_baseline_password=[string]`
   `auth_updated_username=[string]`
   `auth_updated_password=[string]`

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** `{ id : 123abc }`
 
* **Error Response:**

  * **Code:** 406 <br />
    **Content:** `{ 'Error': 'Failed to launch crawler' }`

* **Sample Call:**
  `
    arr = {
      "baseline_url": "https://www.test.com",
      "updated_url": "https://www.test.com",
      "max_depth": "1",
      "max_urls": "1",
      "prefix": "test"
    };
    $.ajax({
        type: "POST",
        url: "/api/verify",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify(arr),
    });
  `

**Verify**
----

* **URL**

  _/api/verify_

* **Method:**

  `GET`

 
* **URL Params**
    **Required:**

    _id: 123abc_

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** `{ Status: In progress }`
  
  OR 

  * **Code:** 200 <br />
    **Content:** `{ Status: Done }`

  OR

  * **Code:** 200 <br />
    **Content:** `{ Status: _status code_ }`
    where _status code_ is 
      'Invalid website',
      'Prediction script failed',
      'Failed to launch inference process',
      'Model does not exist',
      'CHROMEDRIVER_PATH not provided in env variables'

* **Error Response:**

  * **Code:** 404 <br />
    **Content:** `{ 'Error': 'Invalid ID' }`

* **Sample Call:**
  `
    $.ajax({
        type: "GET",
        url: "/api/verify?id=12345"
    });
  `

**ID List**
----

* **URL**

  _/api/ids_

* **Method:**

  `GET`

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** `"id1": {'baseline_url': "http://www.test.com", 
                   'updated_url': "http://www.test.com", 
                   'status': 'Starting',
                   'started_at': get_time(), 
                   'stopped_at': 'None', 
                   'prefix': prefix, 
                   'max_depth': "1",
                   'max_urls': "100"}, 
                   "id2": {'baseline_url': "http://www.test.com", 
                   'updated_url': "http://www.test.com", 
                   'status': 'Starting',
                   'started_at': get_time(), 
                   'stopped_at': 'None', 
                   'prefix': prefix, 
                   'max_depth': "1",
                   'max_urls': "100"}`

* **Sample Call:**
  `
    $.ajax({
        type: "GET",
        url: "/api/ids"
    });
  `

**Results**
----

* **URL**

  _/api/result_

* **Method:**

  `GET`

 
* **URL Params**
    **Required:**

    _id: 123abc_

* **Success Response:**
  
  * **Code:** 200 <br />
    **Content:** `"1": {
        "ui_stats": {
            "mask_div": {
                "images": 0.442,
                "textblock": 0.11,
                "section": 0.034,
                "buttons": 0.0019,
                "forms": 0.037,
                "overall": 0.276
            },
            "pixelwise_div": {
                "images": 0.442,
                "textblock": 0.11,
                "section": 0.4839,
                "buttons": 0.0019,
                "forms": 0.037,
                "overall": 0.276
            },
            "risk_score": 0.276
        },
        "js_stats": {
            "in_baseline": 8,
            "in_upgraded": 1,
            "in_both": 0
        },
        "network_stats": {
            "in_baseline": 100,
            "in_upgraded": 4,
            "in_both": 0
        },
        "risk_score": 1.0,
        "links": {
            "baseline": "https://:@www.test1.com",
            "updated": "https://:@www.test2.com"",
            "endpoint": "/endpoint1",
            "baseline_assets": "tmp/prefix_baseline/",
            "updated_assets": "tmp/prefix_updated/"
        }
    }`

* **Error Response:**

  * **Code:** 404 <br />
    **Content:** `{ 'Error': 'Report does not exist' }`

* **Sample Call:**
  `
    $.ajax({
        type: "GET",
        url: "/api/result?id=12345"
    });
  `


**Results**
----

* **URL**

  _/api/shutdown_

* **Method:**

  `GET`

* **Sample Call:**
  `
    $.ajax({
        type: "GET",
        url: "/api/shutdown"
    });
  `

