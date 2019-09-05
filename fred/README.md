**Verify**
----

* **URL**

  _/verify_

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

  _/verify_

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
        url: "/api/verify?id=12345",
    });
  `
