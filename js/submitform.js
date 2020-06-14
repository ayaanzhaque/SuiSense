const addToFirebase = async (ref, obj) => {
  const { address } = obj;
  const { lat, lng } = await toCoords(address);
  const { key } = ref.push(obj);

  const geoFire = new geofire.GeoFire(ref);
  await geoFire.set(`geo${key}`, [lat(), lng()]);
};

const submit = async (section) => {
  const name = $(`${section} #name`).val();
  const email = $(`${section} #email`).val();
  const address = $(`${section} #address`).val();
  const subject = $(`${section} #subject`).val();
  const message = $(`${section} #message`).val();

  await addToFirebase(refMarkers, { name, email, address, subject, message });
};

const submit2 = async (section) => {
  const name = $(`${section} #name2`).val();
  const email = $(`${section} #email2`).val();
  const address = $(`${section} #address2`).val();
  const phone = $(`${section} #phone`).val();

  await addToFirebase(refVolunteers, { name, email, address, phone });
};

$('#formbutton').click(async (e) => {
  e.preventDefault();
  console.log('IN SUBMIT');
  await submit('#input');
  location.reload();
});

$('#formbuttonE').click(async (e) => {
  e.preventDefault();
  console.log('IN SUBMIT');
  await submit('#inputt');
  location.reload();
});

$('#formbutton2').click(async (e) => {
  e.preventDefault();
  console.log('IN SUBMIT');
  await submit2('#input2');
  location.reload();
});

$('#searchForm').submit((e) => {
  e.preventDefault();
  search('#map2');
});

$('#searchbutton').click((e) => {
  e.preventDefault();
  console.log('IN SUBMIT');
  // $('.nearby-volunteers-list').show();
  // $('.nearby-volunteers-loading-spinner').show();
  search('#map2');
});

$('#searchForm2').submit((e) => {
  e.preventDefault();
  const val = $(`${'#searchvol'} #address3`).val();
  if (val.indexOf(',') != -1 && val.indexOf(',', val.indexOf(',') + 1) != -1) {
    document.getElementById('demo').innerHTML = ' ';
    search('#searchvol');
  } else {
    document.getElementById('demo').innerHTML =
      'Please enter address in the format specified above.';
  }
});

$('#searchbutton2').click((e) => {
  e.preventDefault();
  console.log('IN SUBMIT');
  // $('.nearby-volunteers-list').show();
  // $('.nearby-volunteers-loading-spinner').show();
  const val = $(`${'#searchvol'} #address3`).val();
  if (val.indexOf(',') != -1 && val.indexOf(',', val.indexOf(',') + 1) != -1) {
    document.getElementById('demo').innerHTML = ' ';
    search('#searchvol');
  } else {
    document.getElementById('demo').innerHTML =
      'Please enter address in the format specified above.';
  }
});
