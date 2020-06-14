function sendEmail(host, username, password, to, from, subject, body) {
Email.send({
    Host : host,
    Username : username,
    Password : password,
    To : to,
    From : from,
    Subject : subject,
    Body : body
}).then(
  message => alert(message)
);
}
