## Mac ChromeDriver Installation Instructions
### Install ChromeDriver
Check the latest version of Chrome in your Chrome settings. Download the corresponding ChromeDriver and install it in this directory:

`/usr/local/bin/chromedriver`

We can do so by moving the downloaded chromedriver to the right location in terminal:

`mv --interactive <yourHomeDirPath>/Downloads/chromedriver /usr/local/bin/chromedriver`

The following command can be used, but may install the wrong version of ChromeDriver:
`cdm = ChromeDriverManager().install()`

Once installed, verify that it was installed in the correct location:

`which chromedriver`

### Allowing Permissions to Run
In your Mac Terminal, add permissions to run the ChromeDriver:

`xattr -d com.apple.quarantine /usr/local/bin/chromedriver`

If the above command does not work, try the following:

`spctl --add --label 'Approved' <name-of-executable>`